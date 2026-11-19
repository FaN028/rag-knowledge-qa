# ============================================================
# app.py —— 程序入口：配置页面 + AI 调用 + 初始化状态 + 主程序
# ============================================================
import streamlit as st
import config

# set_page_config 必须是第一个 Streamlit 命令。
# 注意它不再需要抢在"导入模型"之前：模型改成懒加载（embedding.py），
# 导入任何模块都不会加载权重，import 顺序不再是隐式契约。
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

import os
import uuid
from typing import Iterator

from openai import OpenAI

from session import generate_session_name, save_session
import embedding
import knowledge
import retrieval
from agent import run_agent, save_trace
import ui


def build_system_prompt(nick_name: str, nature: str) -> str:
    """根据用户设置生成系统提示词。"""
    return f"""
你叫 {nick_name}，现在是用户的知识库助手，请完全代入知识库助手角色。
规则：
    1. 每次只回1条消息
    2. 禁止任何场景或状态描述性文字
    3. 匹配用户的语言
    4. 回复简短，像微信聊天一样
    5. 有需要的话可以用emoji表情
    6. 用符合知识库助手设定的方式对话
    7. 回复的内容要充分体现知识库助手的性格特征
知识库助手性格：
    - {nature}
你必须严格遵守上述规则来回复用户。
""".strip()


def create_client() -> OpenAI | None:
    """创建 DeepSeek API 客户端。"""
    api_key = os.getenv(config.DEEPSEEK_API_KEY_ENV)

    if not api_key:
        st.error(
            f"没有找到环境变量 {config.DEEPSEEK_API_KEY_ENV}。"
            "请先配置 DeepSeek API Key。"
        )
        return None

    return OpenAI(
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL,
    )


def stream_chat_response(client: OpenAI, messages: list[dict]) -> Iterator[str]:
    """调用模型，并逐段返回 AI 回复。"""
    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=messages,
        stream=True,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    for chunk in response:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            yield content


def run_agent_turn(client: OpenAI, prompt: str, system_prompt: str, history: list[dict]) -> None:
    """Agent 模式下执行一轮对话：调用 agent.run_agent 并渲染结果。

    与「知识库模式」的差异：
      1. 不预先检索知识库，是否检索由模型自行决定；
      2. 无流式输出，需等待全部工具调用结束后一次性返回。
    """
    try:
        with st.spinner("模型正在思考、按需调用工具…"):
            answer, trace = run_agent(client, prompt, system_prompt, history)
    except Exception as exc:
        st.error(f"Agent 调用失败：{exc}")
        return

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_session()

    # 轨迹落盘，供事后复盘与问题定位。
    save_trace(prompt, trace)

    # 工具调用轨迹面板（与「参考来源」面板同构）。
    if trace:
        with st.expander(f"🔧 工具调用轨迹（{len(trace)} 次）"):
            for step in trace:
                st.markdown(f"**第 {step['round']} 轮 · `{step['tool']}`**")
                st.caption(f"参数：{step['args']}")
                st.code(str(step["result"])[:800], language=None)


def init_session_state() -> None:
    """初始化程序运行期间需要保存的数据。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "nick_name" not in st.session_state:
        st.session_state.nick_name = config.DEFAULT_NICK_NAME

    if "nature" not in st.session_state:
        st.session_state.nature = config.DEFAULT_NATURE

    if "current_session" not in st.session_state:
        st.session_state.current_session = generate_session_name()

    # —— 知识库：每个界面会话一份内存库 ——
    # 以前数据的"真相"是 session_state 里的两个 list，检索要另写一遍余弦；
    # 现在直接用一个 KnowledgeBase（和 HTTP 接口同一个类、同一段检索代码），
    # 只是不落盘：访客之间互相看不见，刷新即清空。
    if "kb" not in st.session_state:
        st.session_state.kb = knowledge.session_store(f"ui_{uuid.uuid4().hex[:8]}")
    if "memory" not in st.session_state:
        st.session_state.memory = knowledge.session_store(f"mem_{uuid.uuid4().hex[:8]}")

    # 纯界面状态：上传去重用的文件指纹、清空后刷新上传组件的计数器
    if "knowledge_ids" not in st.session_state:
        st.session_state.knowledge_ids = set()
    if "knowledge_clear_counter" not in st.session_state:
        st.session_state.knowledge_clear_counter = 0

    # —— 对话模式：知识库模式（默认，和原先一样）/ Agent 模式（模型自己决定调不调工具）——
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = config.MODE_KB


def main() -> None:
    """程序入口。"""
    st.title(config.APP_TITLE)

    init_session_state()

    # 告诉工具层：这次脚本运行用的是本会话的知识库。
    # Agent 模式的 search_knowledge 靠它看到用户刚上传的文档——
    # 重构前它去读 session_state，一旦被 HTTP 调用就查不到同一份数据。
    knowledge.bind_store(st.session_state.kb)

    if config.LOGO_PATH.exists():
        st.logo(str(config.LOGO_PATH))

    ui.render_chat_history()
    ui.render_sidebar()

    client = create_client()
    if client is None:
        return

    prompt = st.chat_input("请输入您要问的问题")
    if not prompt:
        return

    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    # 本轮查询向量只算一次：记忆检索、知识库检索、记忆入库共用同一个向量。
    # 重构前一轮对话 encode 三次（记忆入库一次、知识库一次、记忆一次）。
    query_vector = embedding.encode(prompt)

    system_prompt = build_system_prompt(
        st.session_state.nick_name,
        st.session_state.nature,
    )

    # Agent 模式：本轮交由 agent.run_agent 处理，处理完直接返回。
    # history 排除本轮提问——它作为 user_query 单独传入，否则会重复。
    if st.session_state.chat_mode == config.MODE_AGENT:
        st.session_state.memory.add([prompt], [query_vector])
        run_agent_turn(client, prompt, system_prompt, st.session_state.messages[:-1])
        return

    # —— 知识库模式 ——
    # 记忆与知识库两批候选来自不同语料，但 query 是同一个，所以拼成一批
    # 只做一次粗排 + 一次精排；分两次调要付两遍 cross-encoder 的编码开销。
    # 记忆必须在"存入本句"之前查：先存再查的话，本句与自己的相似度是 1.0，
    # 每次都会占掉一个召回名额，等于少检索一条真实的历史。
    memory_hits, kb_hits = retrieval.retrieve_many(
        prompt,
        [st.session_state.memory.search, st.session_state.kb.search],
        query_vector=query_vector,
    )
    memory_context = retrieval.format_context(
        memory_hits,
        instruction=retrieval.MEMORY_INSTRUCTION,
    )
    st.session_state.memory.add([prompt], [query_vector])

    context = retrieval.format_context(kb_hits)
    cited = kb_hits

    if memory_context:
        system_prompt = system_prompt + "\n\n" + memory_context
    if context:
        system_prompt = system_prompt + "\n\n" + context

    request_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *st.session_state.messages,
    ]

    response_placeholder = st.empty()
    full_response = ""

    try:
        for content in stream_chat_response(client, request_messages):
            full_response += content
            with response_placeholder.container():
                with st.chat_message("assistant"):
                    st.write(full_response)
    except Exception as exc:
        st.error(f"调用 AI 失败：{exc}")
        return

    if full_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })
        save_session()

        # 显示"引用来源"
        if cited:
            with st.expander("📚 参考来源"):
                for i, chunk in enumerate(cited, start=1):
                    st.write(f"【片段 {i}】{chunk}")
    else:
        st.warning("AI 没有返回可显示的内容。")


if __name__ == "__main__":
    main()
