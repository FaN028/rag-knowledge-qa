# ============================================================
# embedding.py —— 模型加载（embedding 与 rerank）
#
# 设计要点：
#   1. 懒加载：模块 import 时**不**加载任何模型。切块、鉴权、健康检查
#      这些路径不需要几百 MB 的权重，不应该为它们付启动代价。
#   2. 单例：模型进程内只加载一次。原来用 @st.cache_resource 达到同样
#      效果，但那让核心检索模块依赖 Streamlit——FastAPI 侧根本没法用，
#      只能靠"延迟导入 + try/except"绕。模块级单例谁都能用。
#   3. 加锁：Streamlit 每个会话一个线程、FastAPI 的 def 接口跑在线程池里，
#      两个线程同时首次调用会各自加载一份（2GB × 2）。单把锁就够了。
# ============================================================
import threading

import numpy as np

import config

_model = None
_reranker = None
_lock = threading.Lock()


def get_model():
    """取 embedding 模型（首次调用时加载）。"""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(config.MODEL_PATH)
    return _model


def get_reranker():
    """取 rerank 模型；RAG_RERANK=0 时返回 None（只做向量粗排）。

    rerank 模型常驻约 1.5GB。低配机器（2G 云主机、免费 CPU 实例）会被
    它压垮，所以关掉精排这个选项保留着：精度略降，内存省一半。
    """
    global _reranker
    if not config.RERANK_ENABLED:
        return None
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(config.RERANK_MODEL_PATH)
    return _reranker


def encode(texts):
    """把一段或多段文本编码成向量。

    返回 numpy 数组：单条是 (D,) ，多条是 (N, D)。
    统一转成数组而不是 list，是因为下游（余弦计算、落盘）都要矩阵。
    """
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return np.asarray(vectors, dtype="float32")
