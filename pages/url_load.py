from io import StringIO

import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from app.graph.learning_graph import LearningGraph
from app.tools.loader.file_loader import FileLoader
from loguru import logger

from app.tools.models.model_tokens import models_tokens
from app.tools.models.openai_embeder import OpenaiEmbeder
from app.tools.models.select_llms import get_llms
from app.tools.storage.paradeDB import ParadeDB
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')


st.set_page_config(
    page_title="我的知识库", page_icon="📚", layout="wide"
)


def scrape_url(url):
    learning_graph = LearningGraph(
        config={
            "llm": {
                "model": "gpt-3.5-turbo-ca",
            },
        },
        source=url,
    )
    result = learning_graph.run()
    return result


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str | dict, **kwargs) -> None:
        tokens = token if isinstance(token, str) else token["content"]
        self.text += tokens
        self.container.markdown(self.text)


def scrape_file(file):
    learning_graph = LearningGraph(
        config={
            "llm": {
                "model": "gpt-3.5-turbo-ca",
            },
        },
        source="file",
    )
    result = learning_graph.run()
    return result


all_provider = set(models_tokens.keys())

with st.sidebar:
    url = st.sidebar.text_area(
        "请输入url"
    )
    url_list = url.split("\n")
    sent = st.button(
        "载入:magic_wand:",
    )
    st.sidebar.markdown("-----------")
    st.sidebar.success("选择一个模型.")
    provider = st.sidebar.selectbox(
        "选择一个模型提供商",
        all_provider
    )
    model_name = st.sidebar.selectbox(
        "选择一个模型",
        models_tokens[provider]
    )
    system_prompt = st.sidebar.text_area(
        "系统提示",
        "你是一个能够检索知识并根据检索到知识回答用户问题的机器人"
    )
    set_system_prompt = st.button('重置对话🔁')

    if sent:
        for url in url_list:
            res = scrape_url(url)
            if res != "爬取失败":
                st.success("爬取成功")
            else:
                st.error("爬取失败")
                logger.error(f"{url} 爬取失败")

st.title("🔎 和知识库聊天")

# @traceable
def run_chat():
    if set_system_prompt:
        st.session_state["messages"] = [ChatMessage(role="system", content=system_prompt)]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="system", content=system_prompt)]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        stream_handler = StreamHandler(st.empty())
        llm_config = {
            "provider": provider,
            "model": model_name,
            "streaming": True,
            "callbacks": [stream_handler],
        }
        llm = get_llms(llm_config)

        template = """使用下面的检索到的文章回答我的问题

        {context}
    
        Question: {question}
    
        你的回答:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        db = ParadeDB(
            connection='postgresql+psycopg://pgvector:pgvector@localhost:5432/ai_dev',
            embedding_length=1536,
            embedding_function=OpenaiEmbeder()
        )
        retriever = db.as_retriever(
            search_kwargs={"k": 4}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = rag_chain.invoke(prompt,{"callbacks": [st_cb]})
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))
            # st.write(response)

run_chat()