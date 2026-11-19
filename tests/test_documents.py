# ============================================================
# test_documents.py —— 文档处理纯函数：不加载模型，毫秒级
# ============================================================
from io import BytesIO

import pytest

import config
import documents


# ---------------- chunk_text ----------------

def test_short_text_is_single_chunk():
    assert documents.chunk_text("很短的一句话") == ["很短的一句话"]


def test_whitespace_is_normalized():
    """PDF 抽出来的文字常带一串换行和空格，先压平再切，长度才有意义。"""
    assert documents.chunk_text("第一行\n\n  第二行\t第三行") == ["第一行 第二行 第三行"]


def test_empty_text_yields_no_chunks():
    assert documents.chunk_text("   \n\t ") == []


def test_long_text_is_split_with_overlap():
    text = "".join(str(i % 10) for i in range(1000))

    chunks = documents.chunk_text(text, chunk_size=400, overlap=50)

    assert len(chunks) == 3
    assert [len(c) for c in chunks[:-1]] == [400, 400]
    # 相邻片段共享 overlap 个字符，跨片段的内容才不会在接缝处断掉
    assert chunks[0][-50:] == chunks[1][:50]
    assert chunks[1][-50:] == chunks[2][:50]


def test_defaults_follow_config():
    """默认参数取自 config，改一处即可全局生效。"""
    text = "字" * (config.CHUNK_SIZE + 1)

    chunks = documents.chunk_text(text)

    assert len(chunks) == 2
    assert len(chunks[0]) == config.CHUNK_SIZE


# ---------------- extract_text ----------------

def test_extract_plain_text():
    assert documents.extract_text("note.md", "  # 标题\n正文 ".encode()) == "# 标题\n正文"


def test_extract_rejects_unsupported_type():
    with pytest.raises(documents.DocumentError, match="暂不支持"):
        documents.extract_text("a.xlsx", b"whatever")


def test_extract_rejects_empty_file():
    with pytest.raises(documents.DocumentError, match="没有可用的文字内容"):
        documents.extract_text("empty.txt", b"   ")


def test_extract_docx_paragraphs():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("第一段")
    document.add_paragraph("第二段")
    buffer = BytesIO()
    document.save(buffer)

    assert documents.extract_text("a.docx", buffer.getvalue()) == "第一段\n第二段"


def test_extract_broken_pdf_reports_error():
    pytest.importorskip("pypdf")

    with pytest.raises(documents.DocumentError, match="解析 PDF 失败"):
        documents.extract_text("broken.pdf", b"not a real pdf")
