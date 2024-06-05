import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ChatMessage

from app.tools.models.model_tokens import models_tokens
from app.tools.models.select_llms import get_llms
from app.tools.storage.pg_controller import KnowledgeBaseManager

st.set_page_config(
    page_title="Hello,开始对话吧",
    page_icon="👋",
)

st.write("# Have a nice day! 👋")

all_provider = set(models_tokens.keys())

with st.sidebar:
    st.sidebar.success("选择一个模型.")
    provider = st.sidebar.selectbox(
        "选择一个模型提供商",
        all_provider,
        # "openai" if "openai" in all_provider else list(all_provider)
    )
    model_name = st.sidebar.selectbox(
        "选择一个模型",
        models_tokens[provider],
        # "gpt-4o-ca" if "gpt-4o-ca" in models_tokens[provider] and provider == 'openai' else list(models_tokens[provider])
    )
    system_prompt = st.sidebar.text_area(
        "系统提示",
        "You are a helpful assistant.用中文回答我.如果是代码问题，只需要告诉需要修改的部分的代码"
    )
    set_system_prompt = st.button('重置对话🔁')


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str | dict, **kwargs) -> None:
        tokens = token if isinstance(token, str) else token["content"]
        self.text += tokens
        self.container.markdown(self.text)


if set_system_prompt:
    st.session_state["messages"] = [ChatMessage(role="system", content=system_prompt)]

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="system", content=system_prompt)]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm_config = {
            "provider": provider,
            "model": model_name,
            "streaming": True,
            "callbacks": [stream_handler],
        }

        llm = get_llms(llm_config)
        response = llm.invoke(st.session_state.messages)
        content = response.content if not isinstance(response, str) else response
        st.session_state.messages.append(ChatMessage(role="assistant", content=content.replace("assistant:", "")))


