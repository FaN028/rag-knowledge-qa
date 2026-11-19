# ============================================================
# test_demo.py —— 演示知识库：内容写入、不覆盖已有数据
#                 不加载模型（encoder 由用例注入）
# ============================================================
import demo
import knowledge


def fake_encoder(texts):
    """随便给个向量：这里只验证"写进去了"，不验证语义。"""
    return [[0.1, 0.2, 0.3] for _ in texts]


def test_seed_writes_all_samples():
    kb = knowledge.session_store("demo_test_1")

    total = demo.seed(kb, encoder=fake_encoder)

    assert total == kb.count()
    assert total > 0
    assert sorted(kb.sources) == sorted(demo.SAMPLES)


def test_seed_is_idempotent_and_keeps_user_data():
    kb = knowledge.session_store("demo_test_2")
    kb.add(["用户自己上传的一段话"], [[1.0, 0.0, 0.0]], source="我的笔记.md")

    demo.seed(kb, encoder=fake_encoder)

    # 已有内容时不再写入示例，用户的文档不会被顶掉
    assert kb.count() == 1
    assert kb.sources == ["我的笔记.md"]
