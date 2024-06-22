import asyncio

import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit import rerun

from app.graph import humanin_agent
from app.graph.search_agent import app
from app.tools.models.select_llms import get_llms


class StreamlitAssistantAnswer:
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""

    def re_render_answer(self, token: str) -> None:
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

    def confirm_answer(self, message: str) -> None:
        self.tokens_area.markdown(message)


class GraphConversation:
    def __init__(self, app):
        self.app = app

    async def stream_conversation(self, messages):
        assistant_answer = StreamlitAssistantAnswer()
        async for event in self.app.astream_events(messages, version="v1"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    assistant_answer.re_render_answer(content)

            if kind == "on_chain_end" and event["name"] == "LangGraph":
                # print(event["data"]["output"])
                # message = event["data"]["output"]["__end__"][-1].content
                # assistant_answer.confirm_answer(message)
                # return message
                pass


st.set_page_config(
    page_title="RAG", page_icon="", layout="wide"
)
graph_convo = GraphConversation(app=app)
model = get_llms({"provider": "openai",
    "model": "gpt-4o-ca"})

if user_input_text := st.chat_input("介绍下怎么使用langgraph设计一个可以加入多个智能体的agent"):
    with st.chat_message("user"):
        st.markdown(user_input_text)

    # with st.spinner("回答生成中"):
    callback_handler = StreamlitCallbackHandler(parent_container=st.container(), max_thought_containers=5)

    with st.chat_message("assistant"):
        asyncio.run(graph_convo.stream_conversation(user_input_text,))
