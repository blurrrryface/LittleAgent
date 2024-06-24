import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from openai import BadRequestError

from app.tools.models.model_tokens import models_tokens
from app.tools.models.select_llms import get_llms
from app.tools.storage.pg_controller import KnowledgeBaseManager
import re

all_provider = set(models_tokens.keys())


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


def summary_node(original_text, llm):
    summary_prompt = PromptTemplate.from_template("""
    你是一个专业的文档整体者，接下来你会接受到一个文章的片段，这个片段可能来自与一个项目的文档，也可能来来自于一个科普网页
    你的任务是对这个片段的内容进行总结，比如，你可以根据链接、标题、正文内容对这个文档介绍的内容进行猜测这个文章片段的作用
    每一个总结的长度在400~700的大小,总结的文本要要清晰的条例结构，不要过于简洁
    总结的结果需要是中文，但是对于文章中出现的特有词汇，不要进行翻译
    以下是文章片段：{doc_phrase}""")
    summary_chain = (
            summary_prompt
            | llm
            | StrOutputParser()
    )

    response = summary_chain.invoke({"doc_phrase":original_text})
    return response


def summary(llm):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    manager = KnowledgeBaseManager()
    # 执行查询
    get_num = """
        SELECT count(1) as num
        FROM document.knowledge_base 
        where summary is null;
        """

    get_docs = """
    SELECT id,source as title,url,document_content
    FROM document.knowledge_base 
    where summary is null;
    """
    num = manager.execute_query_one(get_num)["num"]
    docs = manager.execute_query_all(get_docs)
    iterator = iter(docs)
    for i in range(num):
        row = next(iterator)._asdict()

        doc = row["document_content"]
        title = row["title"]
        url = row["url"]
        id = row["id"]

        text = f"""
        文章来源：{url} \n
        文章标题：{title} \n
        正文：\n=====\n{doc} 
        """
        try:
            summary_doc = summary_node(text, llm)
        except BadRequestError as e:
            logger.error(f"总结失败: {id}")
            logger.error(e)
            continue
        manager.update_by_id(id, summary=summary_doc)
        logger.info(f"总结完成: {id}")
        # 更新进度条的值
        progress_bar.progress((i + 1) / num)
        progress_text.text(f"进度: {int((i + 1) / num * 100)}%")


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


def _get_all_doc_title():
    manager = KnowledgeBaseManager()
    get_all_doc_title = """
        SELECT "source"
        FROM document.knowledge_base t1
        group by "source",url
        order by url
        """
    docs_title = manager.execute_query_all(get_all_doc_title)
    return [doc.source for doc in docs_title]


def _get_title_content(doc_title):
    manager = KnowledgeBaseManager()
    get_title_content = f"""
                select links,images,string_agg(document_content,'') as en,string_agg("translation",'') as zh
                from (
                    SELECT links,images,id,document_content,"translation"
                    FROM document.knowledge_base t1
                    where "source" = '{doc_title}'
                    group by id,links,images
                    order by id
                )
                group by links,images
                """
    docs_content = manager.execute_query_one(get_title_content)
    links = docs_content['links']
    images = docs_content['images']
    restored_english_markdown, restored_chinese_markdown = restore_images_in_markdown(
        docs_content['en'], docs_content['zh'], images)
    return restored_english_markdown, restored_chinese_markdown


def restore_images_in_markdown(english_markdown, chinese_markdown, images):
    import re

    # 正则表达式匹配英文Markdown中的图片占位符
    english_image_placeholders = re.findall(r'!\[(.*?)\]\(\)', english_markdown)

    # 替换英文Markdown中的图片链接
    restored_english_markdown = english_markdown
    for placeholder in english_image_placeholders:
        image_url = images.get(placeholder)
        if image_url:
            restored_english_markdown = restored_english_markdown.replace(f"![{placeholder}]()",
                                                                          f"![{placeholder}]({image_url})", 1)

    # 替换中文Markdown中的图片链接
    # 假设图片的顺序与英文Markdown中的顺序相同
    restored_chinese_markdown = chinese_markdown
    for i, placeholder in enumerate(english_image_placeholders):
        image_url = images.get(placeholder)
        if image_url:
            # 查找中文Markdown中的下一个图片占位符
            restored_chinese_markdown = re.sub(r'!\[(.*?)\]\(\)', f'![{placeholder}]({image_url})',
                                               restored_chinese_markdown, 1)

    return restored_english_markdown, restored_chinese_markdown



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
    st.button("开始翻译", on_click=translate, args=(llm,))
    st.button("开始总结", on_click=summary, args=(llm,))

    doc_title = st.sidebar.selectbox(
        "选择一篇文章",
        _get_all_doc_title()
    )

show_souce_content = st.button("查看原文")
show_markdown_content = st.button("查看Markdown")
en_content, zh_content = _get_title_content(doc_title)
st.write("# " + doc_title)
if show_souce_content:
    if show_markdown_content:
        st.text(en_content)
    else:
        st.write(en_content)

else:
    if show_markdown_content:
        st.text(en_content)
    else:
        st.write(zh_content)
