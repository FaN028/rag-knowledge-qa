# ============================================================
# rag.py —— 知识库（RAG）相关：模型加载、切块、解析、检索、注入
# ============================================================
import re
from io import BytesIO

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder

import config


# —— 加载模型（@st.cache_resource = 只加载一次，之后复用）——
@st.cache_resource
def load_model():
    """加载中文 embedding 模型。"""
    return SentenceTransformer(config.MODEL_PATH)


@st.cache_resource
def load_reranker():
    """加载 rerank（重排）模型，用于精排。"""
    return CrossEncoder(config.RERANK_MODEL_PATH)


model = load_model()
reranker = load_reranker()


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """把一段长文本切成若干固定长度、彼此有重叠的小片段。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_text(filename: str, raw: bytes) -> str | None:
    """把上传文件（txt/md/pdf/docx）的原始字节，转成纯文本。"""
    name = filename.lower()

    if name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore").strip() or None

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                st.warning("读取 PDF 需要先安装 pypdf：请在终端运行 `pip install pypdf`。")
                return None
        try:
            reader = PdfReader(BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip() or None
        except Exception as exc:
            st.warning(f"解析 PDF 失败：{exc}")
            return None

    if name.endswith(".docx"):
        try:
            import docx
        except ImportError:
            st.warning("读取 Word 需要先安装 python-docx：请在终端运行 `pip install python-docx`。")
            return None
        try:
            document = docx.Document(BytesIO(raw))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return text.strip() or None
        except Exception as exc:
            st.warning(f"解析 Word 失败：{exc}")
            return None

    st.warning(f"暂不支持的文件类型：{filename}")
    return None


def cosine_similarity(a, b) -> float:
    """算两个向量的余弦相似度，越接近 1 表示意思越像。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_chunks(query: str, chunks: list[str], vectors, top_k: int = config.TOP_K) -> list[str]:
    """两阶段检索：先向量粗排筛出候选，再用 rerank 精排取 top_k。"""
    if not chunks or not vectors:
        return []

    # ① 粗排：向量检索，先筛出前 RERANK_TOP_N 个候选
    q_vec = model.encode(query)
    scored = []
    for idx, chunk in enumerate(chunks):
        score = cosine_similarity(q_vec, vectors[idx])
        scored.append((score, idx, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates = scored[:config.RERANK_TOP_N]

    # ② 精排：rerank 对候选重新打分
    candidate_chunks = [chunk for _, _, chunk in candidates]
    pairs = [[query, chunk] for chunk in candidate_chunks]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(zip(candidate_chunks, rerank_scores), key=lambda item: -item[1])
    return [chunk for chunk, _ in ranked[:top_k]]


def build_context_block(query: str, chunks: list[str], vectors):
    """检索并拼出知识库上下文，同时返回命中的片段（用于显示引用来源）。"""
    selected = retrieve_chunks(query, chunks, vectors)
    if not selected:
        return "", []

    parts = [f"【片段 {i + 1}】\n{chunk}" for i, chunk in enumerate(selected)]
    context = (
        "下面是用户上传的\"知识库\"中，与本次问题最相关的片段：\n"
        "如果用户的问题与这些内容相关，请优先依据这些片段作答；\n"
        "如果无关，请忽略它们，照常以伴侣身份聊天。\n\n"
        + "\n\n".join(parts)
    )
    return context, selected
