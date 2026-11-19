# ============================================================
# tools.py —— Agent 工具集：函数实现 + 注册表 + 声明表
# ============================================================
import re
from datetime import datetime

import config
import knowledge
import retrieval

# ============================================================
# 知识库工具：数据在哪个后端由 knowledge 决定，这里只调接口
#
# 工具层不该知道"界面"和"HTTP"的区别。重构前这里会去读 Streamlit 的
# session_state，失败再 try/except 吞掉——一段代码猜两种运行环境，
# 三种调用方（界面 / Agent / 接口）就各自长出补丁。
# 现在入口层用 knowledge.bind_store() 指明用哪个库，这里只管用。
# ============================================================
def add_knowledge(chunks: list, vectors, source: str = "") -> int:
    """向当前知识库追加片段与向量并立即落盘，返回追加后的片段总数。"""
    store = knowledge.get_store()
    total = store.add(chunks, vectors, source)
    store.save()
    return total


def clear_knowledge() -> None:
    """清空当前知识库（落盘库会一并删掉磁盘数据）。"""
    knowledge.get_store().clear()


def knowledge_info() -> dict:
    """返回当前知识库的状态。"""
    return knowledge.get_store().info()


def search_knowledge(query: str) -> str:
    """在私有知识库中检索与 query 相关的原文片段。

    返回拼接后的原文片段；知识库为空或未命中时返回提示文本。
    """
    store = knowledge.get_store()
    # 先判断是否为空：空库直接返回，避免白白加载几百 MB 的 embedding 模型
    if not store.ready:
        return "知识库中还没有内容。"

    _, selected = retrieval.build_context(query, store.search)
    if not selected:
        return "知识库中没有找到相关内容。"

    # 仅回填原文，不附加任何作答指令，避免工具越权干预模型生成策略。
    return "\n\n".join(f"【片段 {i}】\n{c}" for i, c in enumerate(selected, start=1))


def calculate(expression: str) -> str:
    """计算四则运算表达式并返回字符串结果。

    表达式由模型生成，直接求值存在任意代码执行风险，
    因此依次执行白名单校验、剥离内置函数、异常兜底三层防护。
    """
    # 白名单：整串仅允许数字与 + - * / ( ) 及空白字符。
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "只支持数字和 + - * / ( ) 运算。"

    try:
        # __builtins__ 置空，阻断 import / open / exec 等内置能力。
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"计算失败：{exc}"


def get_current_time() -> str:
    """获取当前日期与时间，格式 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now(config.TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


# 工具注册表：工具名 -> 可调用对象，供调度端按名派发。
TOOLS = {
    "search_knowledge": search_knowledge,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


# 工具声明表：下发至模型的 Function Calling Schema，模型仅可见此表。
# 约束：name 须与 TOOLS 的键一致，parameters 中的参数名须与函数签名一致。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在用户上传的私有知识库中检索相关原文片段。"
                "当用户的问题可能涉及用户上传过的资料、文档、笔记时使用。"
                "返回的是原文片段，不是答案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要在知识库中检索的问题或关键词。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "计算一个数学表达式并返回数值结果。只支持加减乘除和括号。"
                "当用户需要精确算数（四则运算、百分比等）时使用，不要自己心算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 (3+5)*2。",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "获取当前的日期和时间。"
                "当用户问到今天几号、现在几点、当前日期时间时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]
