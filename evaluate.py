# -*- coding: utf-8 -*-
# 阶段4：评测三种检索方法（关键词 / 向量 / 向量+rerank）
# 目标：用「命中率」这个数据，证明哪种检索更好。
# 运行：./.venv/Scripts/python.exe evaluate.py
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

print("加载模型...")
embed_model = SentenceTransformer("models/bge-small-zh-v1.5")
reranker = CrossEncoder("models/bge-reranker-base")
print("完成\n")

# —— 知识库（10 个片段，模拟"公司手册"）——
chunks = [
    "员工每年享有带薪年假5天，须在年底前使用完毕",     # 0
    "报销单需要部门经理签字后提交给财务部",           # 1
    "考勤打卡时间是每天早上九点到下午六点",           # 2
    "绩效评估每半年进行一次，结果影响年终奖金",        # 3
    "加班需要提前一天向主管申请",                     # 4
    "新员工试用期为三个月",                           # 5
    "病假需要提供医院证明",                           # 6
    "出差补贴标准是每天200元",                        # 7
    "离职需要提前一个月通知",                         # 8
    "年终奖在春节前发放",                             # 9
]

# —— 测试集：(问题, 正确答案片段的序号) ——
# 注意：前几题故意用"不同说法"，看关键词能不能搜到
tests = [
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

TOP_K = 3

# 预计算向量
vectors = embed_model.encode(chunks)


# —— 方法1：关键词检索（阶段1的逻辑）——
def tokenize(text):
    text = text.lower()
    terms = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def keyword_retrieve(query):
    qt = set(tokenize(query))
    scored = []
    for idx, c in enumerate(chunks):
        ct = tokenize(c)
        hits = sum(1 for t in qt if t in ct)
        scored.append((hits / (len(ct) ** 0.5), idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for s, idx in scored[:TOP_K] if s > 0]


# —— 方法2：向量检索 ——
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def vector_retrieve(query):
    q = embed_model.encode(query)
    scored = sorted([(cosine(q, v), i) for i, v in enumerate(vectors)], key=lambda x: -x[0])
    return [i for s, i in scored[:TOP_K]]


# —— 方法3：向量 + rerank ——
def rerank_retrieve(query, coarse=6):
    q = embed_model.encode(query)
    scored = sorted([(cosine(q, v), i) for i, v in enumerate(vectors)], key=lambda x: -x[0])
    candidates = scored[:coarse]
    cand_idx = [i for _, i in candidates]
    scores = reranker.predict([[query, chunks[i]] for i in cand_idx])
    ranked = sorted(zip(cand_idx, scores), key=lambda x: -x[1])
    return [i for i, _ in ranked[:TOP_K]]


# —— 算命中率 ——
def hit_rate(retrieve_fn):
    hits = 0
    for question, correct in tests:
        if correct in retrieve_fn(question):
            hits += 1
    return hits / len(tests)


print("=" * 50)
print("测试题数：", len(tests), "，每次检索 Top", TOP_K)
print("=" * 50)
print(f"① 关键词检索   命中率：{hit_rate(keyword_retrieve):.0%}")
print(f"② 向量检索     命中率：{hit_rate(vector_retrieve):.0%}")
print(f"③ 向量 + rerank 命中率：{hit_rate(rerank_retrieve):.0%}")
