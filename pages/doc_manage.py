import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from loguru import logger

from app.tools.models.model_tokens import models_tokens
from app.tools.models.select_llms import get_llms
from app.tools.storage.pg_controller import KnowledgeBaseManager

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

    llm_config = {
        "provider": provider,
        "model": model_name,
        "streaming": False,
    }

    llm = get_llms(llm_config)


def translate_node(original_text, llm):
    translate_prompt = PromptTemplate.from_template("""
    你是一个精通中英文中翻译专家，你的任务是将下面通过标签对<original_text></original_text>包括起来的文章翻译成中文文章。
    在翻译过程中，你需要注意下面几个事项:
    1. 不要破坏文章的原有结构，提供给你的文章都是Markdown结构的，如果你发现文章不完整，可能是切分时截断了
    2. 不要回答我翻译内容以外的任何内容，包括任何解释类的语言
    3. 不要对 `[]()`和 `![]()` 的括号中的内容进行翻译 
    以下是你需要翻译的文章：
    <original_text>
    {original_text}
    </original_text>
    """)

    translate_chain = (
            {"original_text": RunnablePassthrough()}
            | translate_prompt
            | llm
            | StrOutputParser()
    )
    response = translate_chain.invoke(original_text)
    return response


def translate(llm):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    manager = KnowledgeBaseManager()
    # 执行查询
    get_num = """
        SELECT count(1) as num
        FROM document.knowledge_base 
        where translation is null;
        """

    get_docs = """
    SELECT id,document_content
    FROM document.knowledge_base 
    where translation is null;
    """
    num = manager.execute_query_one(get_num)["num"]
    docs = manager.execute_query_all(get_docs)
    iterator = iter(docs)
    for i in range(num):
        row = next(iterator)._asdict()

        doc = row["document_content"]
        id = row["id"]

        trans_doc = translate_node(doc, llm)
        manager.update_by_id(id, translation=trans_doc)
        logger.info(f"翻译完成: {id}")
        # 更新进度条的值
        progress_bar.progress((i + 1) / num)
        progress_text.text(f"进度: {int((i + 1) / num * 100)}%")

st.button("开始翻译", on_click=translate, args=(llm,))
