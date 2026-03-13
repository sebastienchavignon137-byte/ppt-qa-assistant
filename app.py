import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
import openai

# 先从 .env 文件中加载环境变量（本地开发友好）
load_dotenv()

# -------------------------------
# 页面基础配置
# -------------------------------
st.set_page_config(
    page_title="文档智能问答助手",
    page_icon="📄",
    layout="wide",
)


# -------------------------------
# 工具函数：读取 API Key / Base URL
# -------------------------------
def get_api_key() -> str:
    """
    优先从 st.secrets 中获取 API_KEY；
    如果没有配置，再从系统环境变量中读取。
    两者都没有时，返回 None。
    """
    api_key = None

    # 1. 优先从 st.secrets 获取（适合部署到 Streamlit Cloud 等）
    try:
        if "API_KEY" in st.secrets:
            api_key = st.secrets["API_KEY"]
    except Exception:
        # 如果 st.secrets 不可用（极少见），直接忽略异常
        pass

    # 2. 如果上面没有拿到，再尝试从环境变量中获取
    if not api_key:
        api_key = os.getenv("API_KEY")

    return api_key


def get_base_url() -> str:
    """
    读取大模型的 Base URL。
    - 可选配置：用于兼容 DeepSeek、智谱等 OpenAI 接口兼容的服务。
    - 如果没有配置，则返回 None，使用官方默认地址。
    """
    base_url = None

    try:
        if "BASE_URL" in st.secrets:
            base_url = st.secrets["BASE_URL"]
    except Exception:
        pass

    if not base_url:
        base_url = os.getenv("BASE_URL")

    return base_url


# -------------------------------
# 工具函数：初始化 OpenAI 客户端配置
# -------------------------------
def init_openai_client():
    """
    初始化 openai 库所需的配置信息（API Key 和可选的 Base URL）。
    这里使用的是 openai==0.28.x 的经典用法，兼容性较好。
    """
    api_key = get_api_key()
    if not api_key:
        st.error(
            "未检测到 API_KEY，请在 `.env` 或部署平台的 Secrets 中配置 API_KEY。"
        )
        st.stop()

    openai.api_key = api_key

    base_url = get_base_url()
    if base_url:
        # 对于 DeepSeek/智谱等兼容 OpenAI 接口的服务，可以在这里配置自定义地址
        openai.api_base = base_url


# -------------------------------
# 工具函数：从文本文档中读取文本
# -------------------------------
def extract_text_from_text_file(uploaded_file) -> str:
    """
    从上传的文本文档中提取文本内容。
    这里简单地把整个文件内容读出来并解码为字符串。

    支持的典型格式：
    - .txt：纯文本，直接按 UTF-8 解码
    - .md：Markdown 文本，同样按 UTF-8 解码

    注意：
    - 如果你的文件不是 UTF-8 编码，可能需要根据实际情况修改编码方式。
    """
    try:
        # uploaded_file 是一个类似 BytesIO 的对象，可以直接读取 bytes
        raw_bytes = uploaded_file.read()

        # 尝试按 UTF-8 解码
        text = raw_bytes.decode("utf-8", errors="ignore")

        # 去掉首尾多余空白
        return text.strip()
    except Exception as e:
        # 出现任何异常时，在界面上提示，并返回空字符串
        st.error(f"读取文档内容时发生错误：{e}")
        return ""


# -------------------------------
# 工具函数：初始化会话状态
# -------------------------------
def init_session_state():
    """
    初始化 Streamlit 的 session_state 字段：
    - doc_text: 保存整个文档的文本内容
    - doc_file_name: 当前上传的文档文件名
    - messages: 聊天记录（用于多轮对话）
    """
    if "doc_text" not in st.session_state:
        st.session_state["doc_text"] = ""
    if "doc_file_name" not in st.session_state:
        st.session_state["doc_file_name"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # 每个元素为 {"role": "user"/"assistant", "content": "..."}


# -------------------------------
# 工具函数：调用大模型（流式输出）
# -------------------------------
def chat_with_model_stream(user_query: str):
    """
    使用 openai.ChatCompletion.create 实现流式对话。
    - 将文档全文作为 System Prompt 的背景知识
    - 携带历史对话消息，实现多轮记忆
    - 使用 stream=True 获取流式返回，在页面上模拟“打字机”效果
    """
    if not st.session_state["doc_text"]:
        st.warning("请先上传并成功解析一个文本文档，然后再提问。")
        return

    # 组装 System Prompt，其中包含文档文本内容
    system_prompt = (
        "你是一个专业的文档内容讲解助手。下面是用户上传的文本文档内容，"
        "你需要基于这些内容来回答用户问题。如果文档中没有相关信息，"
        "请如实说明，并尽量给出合理的推理或建议。\n\n"
        "【文档内容开始】\n"
        f"{st.session_state['doc_text']}\n"
        "【文档内容结束】"
    )

    # 将 System Prompt 作为第一条消息
    messages = [{"role": "system", "content": system_prompt}]

    # 历史对话：从 session_state 中取出
    messages.extend(st.session_state["messages"])

    # 当前用户提问：追加到消息列表中
    messages.append({"role": "user", "content": user_query})

    # 初始化 OpenAI 客户端配置（API Key / Base URL）
    init_openai_client()

    # 在页面上创建一个“助手消息”容器，用于显示流式内容
    with st.chat_message("assistant"):
        # 这个占位符会在循环中不断被更新，形成打字机效果
        message_placeholder = st.empty()
        full_response = ""

        try:
            # 使用 stream=True 启用流式输出
            # 如果你使用的是 DeepSeek/智谱等兼容服务，请在模型名称和 BASE_URL 中自行调整
            response_stream = openai.ChatCompletion.create(
                model="deepseek-chat",  # 可根据实际服务替换为兼容的模型名称
                messages=messages,
                temperature=0.3,
                stream=True,
            )

            for chunk in response_stream:
                # 每个 chunk 中只包含新增的一小段内容
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    token = delta["content"]
                    full_response += token
                    # 实时将目前累积的文本显示到页面上
                    message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"调用大模型接口时发生错误：{e}")
            return

    # 将完整的助手回复加入到 session_state 中，方便后续多轮对话
    st.session_state["messages"].append({"role": "assistant", "content": full_response})


# -------------------------------
# 主应用逻辑
# -------------------------------
def main():
    # 初始化会话状态
    init_session_state()

    # 页面标题
    st.title("📄 文本智能问答助手")
    st.write(
        "上传一个文本文档（例如 .txt / .md），系统会自动读取其中的文字内容，"
        "然后你可以像 ChatGPT 一样就文档内容进行智能问答。"
    )

    # 侧边栏：文件上传区域
    with st.sidebar:
        st.header("📁 上传文本文档")
        uploaded_file = st.file_uploader(
            "请选择一个文本文档（例如 .txt 或 .md）",
            type=["txt", "md"],
            help="目前仅支持纯文本格式的文件，例如 TXT、Markdown。",
        )

        # 当用户上传了文件时，立即尝试解析并存入 session_state
        if uploaded_file is not None:
            try:
                with st.spinner("正在读取文档文本，请稍候..."):
                    doc_text = extract_text_from_text_file(uploaded_file)
                st.session_state["doc_text"] = doc_text
                st.session_state["doc_file_name"] = uploaded_file.name

                st.success(f"已成功读取文档：{uploaded_file.name}")
                # 展示前几百个字符，帮助用户确认是否解析成功
                preview_len = 500
                if doc_text:
                    st.caption("以下是从文档中提取出的部分文本预览：")
                    st.text(
                        doc_text[:preview_len]
                        + ("..." if len(doc_text) > preview_len else "")
                    )
                else:
                    st.warning("文档内容为空，请检查文件。")

            except Exception as e:
                st.error(f"读取文档文件时发生错误：{e}")

        else:
            st.info("请在这里上传一个文本文档，以便开始问答。")

    # 主区域：聊天界面
    st.subheader("💬 基于文档的智能问答")

    # 显示当前已解析的文档文件名
    if st.session_state["doc_file_name"]:
        st.markdown(f"**当前文档：** `{st.session_state['doc_file_name']}`")
    else:
        st.markdown("**当前尚未上传文本文档。**")

    # 将历史对话记录渲染到页面中
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 底部输入框：让用户输入新的问题
    user_query = st.chat_input("请在这里输入你关于文档的任何问题...")

    # 如果用户输入了内容
    if user_query:
        # 先把用户消息显示在对话区，并写入 session_state 中
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 然后调用大模型进行回答（流式输出）
        chat_with_model_stream(user_query)


if __name__ == "__main__":
    main()