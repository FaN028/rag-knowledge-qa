# ============================================================
# knowledge.py —— 服务端知识库：按客户端隔离 + 后端可切换的持久化
# ============================================================
import contextvars
import re
from pathlib import Path

import config
import vectorstore

# 当前这次请求（或这次 Streamlit 脚本运行）该用哪个知识库。
#
# 工具函数（search_knowledge）是被模型调用的，它拿不到 HTTP 请求头，
# 也拿不到 Streamlit 的 session_state。于是由入口层（api.py 的中间件、
# app.py 的 main）在开始处"绑定"好，工具函数只读不猜。
# 用 ContextVar 而不是全局变量：FastAPI 并发处理多个请求时，
# 每个请求有独立的上下文，不会互相串库。
_current_kb: contextvars.ContextVar["KnowledgeBase | None"] = contextvars.ContextVar(
    "current_kb", default=None
)


def normalize_client_id(raw: str | None) -> str:
    """把客户端标识规范化为安全的目录名；缺省或非法时归入 default。"""
    if not raw:
        return "default"
    safe = re.sub(r"[^0-9a-zA-Z_-]", "", raw)[:40]
    return safe or "default"


def bind_store(kb: "KnowledgeBase | None") -> None:
    """把「这次请求用哪个知识库」写入上下文（入口层调用）。"""
    _current_kb.set(kb)


def set_current_client(client_id: str | None) -> None:
    """按客户端标识绑定对应的落盘知识库（等价于 bind_store + 查注册表）。"""
    bind_store(get_store(client_id))


def current_client() -> str:
    """读取当前绑定的知识库属于哪个客户端。"""
    kb = _current_kb.get()
    return kb.client_id if kb is not None else "default"


class KnowledgeBase:
    """单个客户端的知识库。

    只负责「按客户端隔离」这一层语义，真正的存取与检索交给
    vectorstore 里的后端（NumPy 全量扫描 / Chroma HNSW），
    因此换后端不会牵动这里的任何一行调用代码。

    persistent=False 时是「纯内存库」：数据随对象回收而消失，不落盘。
    界面会话用它——多个访客各自一份，互相看不见，刷新就没了。
    """

    def __init__(self, client_id: str, persistent: bool = True) -> None:
        self.client_id = client_id
        self.persistent = persistent
        # 懒加载：第一次真正用到时才建后端（Chroma 会开连接，代价不为零）
        self._store = None

    # ---------------- 目录与后端 ----------------

    @property
    def directory(self) -> Path:
        return config.KNOWLEDGE_DIR / self.client_id

    @property
    def store(self):
        """按客户端隔离的向量存储实例。

        内存库强制用 numpy：Chroma 的价值在持久化 + 大库 HNSW，
        一个随会话消失的临时库既不需要落盘，也没大到需要建索引。
        """
        if self._store is None:
            self._store = vectorstore.build_store(
                self.client_id,
                self.directory,
                backend=None if self.persistent else "numpy",
            )
        return self._store

    @property
    def backend(self) -> str:
        """当前后端名称：numpy / chroma（便于健康检查与日志）。"""
        return self.store.backend

    # ---------------- 只读视图 ----------------

    @property
    def chunks(self) -> list[str]:
        return self.store.chunks()

    @property
    def vectors(self) -> list:
        return self.store.vectors()

    @property
    def sources(self) -> list[str]:
        return self.store.sources()

    @property
    def ready(self) -> bool:
        """是否已有可检索的内容。"""
        return self.store.count() > 0

    def info(self) -> dict:
        return {"sources": self.sources, "chunks": self.store.count()}

    def count(self) -> int:
        """当前片段总数（健康检查与统计用）。"""
        return self.store.count()

    # ---------------- 增删查 ----------------

    def add(self, chunks: list[str], vectors, source: str = "") -> int:
        """追加片段与向量，返回追加后的片段总数。"""
        return self.store.add(chunks, vectors, source)

    def search(self, query_vector, limit: int) -> list[str]:
        """按向量相似度取候选片段，供上层 rerank 精排。"""
        return self.store.query(query_vector, limit)

    def clear(self) -> None:
        """清空该客户端的知识库（含已落盘数据）。"""
        self.store.clear()

    # ---------------- 持久化 ----------------

    def save(self) -> None:
        """落盘；Chroma 后端自身持久化，这里只保证接口一致。

        纯内存库直接跳过——它的语义就是"不留下痕迹"。
        """
        if not self.persistent:
            return
        self.store.persist()

    def load(self) -> bool:
        """从磁盘恢复；无数据或已损坏时返回 False。"""
        if not self.persistent:
            return False
        return self.store.load()


# ---------------- 客户端 → 知识库 的注册表 ----------------

_STORES: dict[str, KnowledgeBase] = {}


def get_store(client_id: str | None = None) -> KnowledgeBase:
    """取知识库。

    - 不传 client_id：用入口层 bind 进来的那个（工具函数就这么调）；
      没有绑定则退回 default 客户端的落盘库。
    - 传了 client_id：取该客户端的落盘库（跨请求共享、会被持久化）。

    首次访问某个客户端时，会尝试从磁盘恢复其历史数据。
    """
    if client_id is None:
        bound = _current_kb.get()
        if bound is not None:
            return bound
        client_id = "default"

    key = normalize_client_id(client_id)
    if key not in _STORES:
        store = KnowledgeBase(key)
        store.load()
        _STORES[key] = store

    return _STORES[key]


def session_store(key: str) -> KnowledgeBase:
    """新建一个纯内存知识库，给界面会话用。

    刻意**不**注册进 _STORES：Streamlit 每个访客一个会话，放进全局表
    就等于每次刷新都漏一份数据，直到进程退出。让 session_state 持有它，
    会话结束随对象一起回收。
    """
    return KnowledgeBase(normalize_client_id(key), persistent=False)


def forget_all() -> None:
    """清空内存中的注册表（测试用；不影响磁盘数据）。"""
    _STORES.clear()
    bind_store(None)
