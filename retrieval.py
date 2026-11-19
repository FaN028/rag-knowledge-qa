# ============================================================
# retrieval.py —— 检索与上下文拼装（全项目唯一的一条检索路径）
#
# 调用方只需要给一个 "searcher"，也就是 KnowledgeBase.search：
#
#     retrieval.build_context(query, kb.search)
#
# 界面模式和 Agent 模式走的是同一段代码——这正是重构前缺的东西：
# 以前界面模式走 rag.py 里的 Python for 循环 + np.dot，Agent 模式走
# vectorstore.py 的矩阵运算 + argpartition，同分排序规则都不一样，
# 同一份文档在两种模式下可能给出不同的 Top-K。
# ============================================================
import config
import embedding

# 拼进 system prompt 的两段"来源说明"。
# 分开写的理由：这两批片段的性质不同，对模型的要求也不同。
# 以前记忆片段复用了知识库的说明，被模型当成"用户上传的资料"，
# 于是它会拿"上周聊过的一句话"当知识库证据引用——答非所问。
KB_INSTRUCTION = (
    "下面是用户上传的\"知识库\"中，与本次问题最相关的片段：\n"
    "如果用户的问题与这些内容相关，请优先依据这些片段作答；\n"
    "如果无关，请忽略它们，照常以知识库助手身份聊天。"
)

MEMORY_INSTRUCTION = (
    "【你的记忆】下面是本次会话中，与当前问题相关的历史对话片段"
    "（多是用户之前说过的话）：\n"
    "它们只用于保持上下文连贯、弄清\"它/那个/刚才说的\"指的是什么；\n"
    "不要把记忆当作可引用的资料，也不要在回答里复述它们。"
)


def retrieve(
    query: str,
    searcher,
    *,
    query_vector=None,
    top_k: int | None = None,
    rerank: bool = True,
) -> list[str]:
    """两阶段检索：向量粗排取候选 → rerank 精排取 top_k。

    参数：
        searcher      —— searcher(query_vector, limit) -> list[str]，
                         由向量库后端实现（numpy 全量扫描 / Chroma HNSW）。
                         检索只认这个接口，所以换后端不影响这里一行代码。
        query_vector  —— 已经算好的查询向量。一轮对话里知识库与记忆要
                         查两次，传进来就能只 encode 一次（原来 encode 三次）。
        rerank        —— 是否精排。记忆条目短、候选少，可以关掉省一次
                         1.5GB 模型的往返。
    """
    if top_k is None:
        top_k = config.TOP_K
    if searcher is None:
        return []

    if query_vector is None:
        query_vector = embedding.encode(query)

    candidates = list(searcher(query_vector, config.RERANK_TOP_N))
    if not candidates:
        return []

    if not rerank:
        return candidates[:top_k]

    ranker = embedding.get_reranker()
    if ranker is None:  # RAG_RERANK=0：只做向量粗排
        return candidates[:top_k]

    scores = ranker.predict([[query, chunk] for chunk in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda item: -item[1])
    return [chunk for chunk, _ in ranked[:top_k]]


def build_context(
    query: str,
    searcher,
    *,
    query_vector=None,
    instruction: str = KB_INSTRUCTION,
) -> tuple[str, list[str]]:
    """检索并拼出可注入 prompt 的上下文块。

    返回 (上下文文本, 命中的片段)。片段单独返回是为了界面上的
    "参考来源"面板——渲染与投喂是两件事。
    """
    selected = retrieve(query, searcher, query_vector=query_vector)
    if not selected:
        return "", []

    parts = [f"【片段 {i + 1}】\n{chunk}" for i, chunk in enumerate(selected)]
    return instruction + "\n\n" + "\n\n".join(parts), selected
