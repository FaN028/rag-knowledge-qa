# -*- coding: utf-8 -*-
# ============================================================
# evaluate.py —— 评测三种检索方法的命中率
# ============================================================
"""目标：用「命中率」这个数据，证明哪种检索更好。

两种用法：

    python evaluate.py          # 打印对比表，给人看结果
    pytest -m slow              # 作为回归测试跑（见 tests/test_retrieval_quality.py）

**这里测的是生产链路**：向量检索与精排都直接调 retrieval.retrieve +
NumpyVectorStore，和界面 / Agent 模式跑的是同一段代码。
早期的版本自己重写了一遍余弦召回，评测结果与实际运行效果可能对不上，
那样的数字没有意义——测的不是会部署的那份实现。

关键词检索是唯一保留的"平行实现"：它是本评测的对照组基线，
本来就不在生产链路里（生产不做关键词检索）。

模型按需加载、不放在模块顶层：测试收集阶段会 import 本模块，
若在导入时就加载 1.2G 模型，默认的快速测试（8 秒）会被拖成一分多钟。
"""
import re
from functools import lru_cache
from pathlib import Path

import config
import embedding
import retrieval
import vectorstore

TOP_K = 3

# —— 知识库（10 个片段，模拟"公司手册"）——
CHUNKS = [
    "员工每年享有带薪年假5天，须在年底前使用完毕",     # 0
    "报销单需要部门经理签字后提交给财务部",           # 1
    "考勤打卡时间是每天早上九点到下午六点",           # 2
    "绩效评估每半年进行一次，结果影响年终奖金",       # 3
    "加班需要提前一天向主管申请",                     # 4
    "新员工试用期为三个月",                           # 5
    "病假需要提供医院证明",                           # 6
    "出差补贴标准是每天200元",                        # 7
    "离职需要提前一个月通知",                         # 8
    "年终奖在春节前发放",                             # 9
]

# —— 测试集：(问题, 正确答案片段的序号) ——
# 注意：前几题故意用"不同说法"，看关键词能不能搜到
TESTS = [
    ("一年能休息多少天", 0),     # "年假" vs "休息" —— 字不同
    ("报销找谁签字", 1),
    ("几点开始上班", 2),         # "上班" vs "打卡" —— 字不同
    ("多久考核一次", 3),         # "考核" vs "评估" —— 字不同
    ("加班要提前申请吗", 4),
    ("试用期多久", 5),
    ("生病请假要什么证明", 6),   # "生病" vs "病假" —— 字不同
    ("出差每天补贴多少", 7),
    ("辞职要提前多久说", 8),     # "辞职" vs "离职" —— 字不同
    ("年终奖什么时候发", 9),
]

_CHUNK_INDEX = {chunk: i for i, chunk in enumerate(CHUNKS)}


@lru_cache(maxsize=1)
def eval_store() -> vectorstore.NumpyVectorStore:
    """把 10 个片段装进真实的向量库后端，供生产检索链路查询。

    不落盘：目录只是构造参数，NumpyVectorStore 只在 persist() 时才写盘。
    """
    store = vectorstore.NumpyVectorStore(Path(config.KNOWLEDGE_DIR) / "_evaluate")
    store.add(list(CHUNKS), embedding.encode(CHUNKS), source="evaluate.py")
    return store


def _to_indices(chunks: list[str]) -> list[int]:
    return [_CHUNK_INDEX[chunk] for chunk in chunks]


# —— 对照组：关键词检索（生产不使用，仅用于对比）——
def tokenize(text: str) -> list[str]:
    text = text.lower()
    terms = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def keyword_retrieve(query: str, top_k: int = TOP_K) -> list[int]:
    query_terms = set(tokenize(query))
    scored = []
    for idx, chunk in enumerate(CHUNKS):
        chunk_terms = tokenize(chunk)
        hits = sum(1 for t in query_terms if t in chunk_terms)
        scored.append((hits / (len(chunk_terms) ** 0.5), idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for score, idx in scored[:top_k] if score > 0]


# —— 方法2：向量检索（走生产链路）——
def vector_retrieve(query: str, top_k: int = TOP_K) -> list[int]:
    chunks = retrieval.retrieve(
        query, eval_store().query, top_k=top_k, rerank=False
    )
    return _to_indices(chunks)


# —— 方法3：向量 + rerank（走生产链路）——
def rerank_retrieve(query: str, top_k: int = TOP_K) -> list[int]:
    chunks = retrieval.retrieve(query, eval_store().query, top_k=top_k)
    return _to_indices(chunks)


def hit_rate(retrieve_fn, top_k: int = TOP_K) -> float:
    """命中率：正确答案是否落在 Top-K 结果里。"""
    hits = sum(1 for question, correct in TESTS if correct in retrieve_fn(question))
    return hits / len(TESTS)


def main() -> None:
    print("加载模型...")
    eval_store()
    print("完成\n")

    print("=" * 50)
    print("测试题数：", len(TESTS), "，每次检索 Top", TOP_K)
    print("=" * 50)
    print(f"1) 关键词检索   命中率：{hit_rate(keyword_retrieve):.0%}   （对照组）")
    print(f"2) 向量检索     命中率：{hit_rate(vector_retrieve):.0%}")
    print(f"3) 向量 + rerank 命中率：{hit_rate(rerank_retrieve):.0%}")
    if not config.RERANK_ENABLED:
        print("\n注意：RAG_RERANK=0，第 3 行与第 2 行相同（精排已关闭）。")


if __name__ == "__main__":
    main()
