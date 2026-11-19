# ============================================================
# test_retrieval.py —— 检索链路：只有一条路，且不依赖 Streamlit
# ============================================================
import pathlib

import pytest

import config
import embedding
import knowledge
import retrieval

CANDIDATES = ["片段甲", "片段乙", "片段丙"]


class FakeRanker:
    """假的 cross-encoder：按预设分数打分，不加载 1.5G 模型。"""

    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return self._scores[:len(pairs)]


@pytest.fixture(autouse=True)
def block_real_models(monkeypatch):
    """兜底：本文件任何用例都不许加载真实模型。

    一次误加载就是 1.5G 内存 + 100 秒，而且会静默拖慢整个测试套件——
    与其靠每个用例自觉，不如让"忘了关"直接失败。需要替身的用例
    在自己的 fixture 里覆盖这两个名字即可。
    """

    def boom(*_args, **_kwargs):
        raise AssertionError("测试里不该加载真实模型")

    monkeypatch.setattr(embedding, "get_model", boom)
    monkeypatch.setattr(embedding, "get_reranker", boom)


@pytest.fixture
def no_encoder(monkeypatch):
    """禁止真正 encode：一旦被调用就让用例失败，并记下次数。"""

    def boom(*_args, **_kwargs):
        raise AssertionError("不应该在这里 encode")

    monkeypatch.setattr(embedding, "encode", boom)


@pytest.fixture
def no_rerank(monkeypatch):
    """关掉精排：RAG_RERANK=0 的等价形态。"""
    monkeypatch.setattr(embedding, "get_reranker", lambda: None)


def fake_searcher(candidates=None, record=None):
    """造一个 searcher(query_vector, limit)，和 KnowledgeBase.search 同签名。"""

    def search(query_vector, limit):
        if record is not None:
            record.append((query_vector, limit))
        return list(CANDIDATES if candidates is None else candidates)

    return search


# ---------------- 粗排 + 精排 ----------------

def test_rerank_reorders_candidates(monkeypatch, no_encoder):
    monkeypatch.setattr(embedding, "get_reranker", lambda: FakeRanker([0.1, 0.9, 0.5]))

    result = retrieval.retrieve("问题", fake_searcher(), query_vector=[0.0], top_k=2)

    assert result == ["片段乙", "片段丙"]  # 按 rerank 分数降序，不是粗排顺序


def test_rerank_off_returns_coarse_order(no_encoder, no_rerank):
    result = retrieval.retrieve("问题", fake_searcher(), query_vector=[0.0], top_k=2)

    assert result == ["片段甲", "片段乙"]


def test_coarse_limit_comes_from_config(no_encoder, no_rerank):
    record = []

    retrieval.retrieve("问题", fake_searcher(record=record), query_vector=[0.0])

    # 粗排只取 RERANK_TOP_N 个候选；把全库都送去 rerank 会白白变慢
    assert record[0][1] == config.RERANK_TOP_N


# ---------------- 只 encode 一次 ----------------

def test_passed_vector_skips_encoding(no_encoder, no_rerank):
    """一轮对话里知识库与记忆共用同一个查询向量。"""
    retrieval.retrieve("问题", fake_searcher(), query_vector=[0.0])


def test_encode_happens_once_when_no_vector(monkeypatch):
    calls = []

    def counting_encode(texts):
        calls.append(texts)
        return [0.0]

    monkeypatch.setattr(embedding, "encode", counting_encode)
    monkeypatch.setattr(embedding, "get_reranker", lambda: None)

    retrieval.build_context("问题", fake_searcher())

    assert calls == ["问题"]


# ---------------- 空库与无候选 ----------------

def test_empty_candidates_return_nothing(no_encoder):
    assert retrieval.retrieve("问题", fake_searcher([]), query_vector=[0.0]) == []
    assert retrieval.build_context("问题", fake_searcher([]), query_vector=[0.0]) == ("", [])


def test_missing_searcher_returns_nothing(no_encoder):
    assert retrieval.retrieve("问题", None, query_vector=[0.0]) == []


# ---------------- 上下文拼装 ----------------

def test_context_contains_instruction_and_numbered_parts(no_encoder, no_rerank):
    context, selected = retrieval.build_context(
        "问题", fake_searcher(), query_vector=[0.0], instruction="【说明】"
    )

    assert selected == CANDIDATES[: config.TOP_K]
    assert context.startswith("【说明】")
    assert "【片段 1】" in context and "片段甲" in context


def test_memory_instruction_is_not_the_knowledge_instruction():
    """记忆片段不能被说成"用户上传的知识库"——否则模型会把它当资料引用。"""
    assert "知识库" in retrieval.KB_INSTRUCTION
    assert "知识库" not in retrieval.MEMORY_INSTRUCTION
    assert "记忆" in retrieval.MEMORY_INSTRUCTION


def test_memory_context_uses_its_own_instruction(no_encoder, no_rerank):
    context, _ = retrieval.build_context(
        "问题",
        fake_searcher(),
        query_vector=[0.0],
        instruction=retrieval.MEMORY_INSTRUCTION,
    )

    assert retrieval.MEMORY_INSTRUCTION in context
    assert retrieval.KB_INSTRUCTION not in context


# ---------------- 与内存知识库串起来 ----------------

def test_session_store_works_with_retrieval(monkeypatch):
    """界面用的内存库 + 生产检索链路要能直接串起来（这就是 app.py 的用法）。"""
    monkeypatch.setattr(embedding, "encode", lambda _query: [1.0, 0.0])
    monkeypatch.setattr(embedding, "get_reranker", lambda: None)

    kb = knowledge.session_store("ui_test")
    kb.add(["苹果"], [[1.0, 0.0]])
    kb.add(["香蕉"], [[0.0, 1.0]])

    _context, selected = retrieval.build_context("水果", kb.search)

    # 候选不足 top_k 时全部返回（不做分数阈值过滤），最相关的排在最前
    assert selected[0] == "苹果"
    assert kb.persistent is False
    # 内存库不该进全局注册表，否则每个访客都漏一份到进程结束
    assert "ui_test" not in knowledge._STORES


# ---------------- 工程约束 ----------------

# 界面侧的三个文件是这个规则唯一允许的例外：它们本来就只做渲染和会话状态
UI_MODULES = {"app.py", "ui.py", "session.py"}


def test_only_ui_modules_import_streamlit():
    """依赖方向要能自动检查：除界面侧外，任何模块都不许 import streamlit。

    这条规则是「同一套逻辑能被 FastAPI 和评测脚本复用」的前提——
    破了它就是又回到了「import 顺序成了隐式契约」。
    """
    root = pathlib.Path(retrieval.__file__).parent
    offenders = []

    for path in sorted(root.glob("*.py")):
        if path.name in UI_MODULES:
            continue
        source = path.read_text(encoding="utf-8")
        if "import streamlit" in source:
            offenders.append(path.name)

    assert offenders == [], f"这些模块不该依赖 Streamlit：{offenders}"
