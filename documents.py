# ============================================================
# documents.py —— 文档处理：上传文件解析 + 文本切块
#
# 这里只有纯函数：不加载模型、不认识 Streamlit、不碰向量库，
# 因此可以脱离整个系统单独测试（见 tests/test_documents.py）。
# ============================================================
import re
from io import BytesIO

import config

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf", ".docx")


class DocumentError(RuntimeError):
    """无法从上传文件中取出文字：格式不支持、解析失败或内容为空。

    消息会直接显示给用户，所以写成"用户看得懂的话"而不是异常堆栈。
    """


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """把一段长文本切成若干固定长度、彼此有重叠的小片段。

    先做空白归一化：PDF 抽出来的文字常常一串换行和多余空格，
    不先压平会让切块长度在"字符数"上失真。
    """
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


def extract_text(filename: str, raw: bytes) -> str:
    """把上传文件（txt/md/pdf/docx）的原始字节转成纯文本。

    失败一律抛 DocumentError，调用方（界面 / 接口）负责决定怎么提示；
    这样解析逻辑不用知道自己是跑在 Streamlit 里还是 HTTP 里。
    """
    name = filename.lower()

    if name.endswith((".txt", ".md")):
        text = raw.decode("utf-8", errors="ignore").strip()
        if not text:
            raise DocumentError(f"《{filename}》没有可用的文字内容。")
        return text

    if name.endswith(".pdf"):
        text = _extract_pdf(raw)
    elif name.endswith(".docx"):
        text = _extract_docx(raw)
    else:
        raise DocumentError(
            f"暂不支持的文件类型：{filename}（仅支持 {' / '.join(SUPPORTED_SUFFIXES)}）。"
        )

    if not text.strip():
        raise DocumentError(f"《{filename}》没有可用的文字内容。")
    return text.strip()


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # 老环境可能只装了 PyPDF2
        except ImportError as exc:
            raise DocumentError(
                "读取 PDF 需要先安装 pypdf：pip install pypdf"
            ) from exc

    try:
        reader = PdfReader(BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pypdf 对坏文件抛的异常类型很杂，统一收口
        raise DocumentError(f"解析 PDF 失败：{exc}") from exc


def _extract_docx(raw: bytes) -> str:
    try:
        import docx
    except ImportError as exc:
        raise DocumentError(
            "读取 Word 需要先安装 python-docx：pip install python-docx"
        ) from exc

    try:
        document = docx.Document(BytesIO(raw))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        raise DocumentError(f"解析 Word 失败：{exc}") from exc
