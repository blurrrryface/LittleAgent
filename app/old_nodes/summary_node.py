from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.old_nodes.base_node import BaseNode


class SummaryNode(BaseNode):

    def _format_documents(self, docs: List[Document]) -> List[str]:
        doc_list = []
        for doc in docs:
            title = doc.metadata.get("source")
            url = doc.metadata.get("url")
            content = doc.page_content

            doc_list.append(f'===\n这篇文章的标题是{title}\n文章的来源为：{url}\n以下是文章的正文：\n{content}\n===')

        return doc_list

    def execute(self, state: dict) -> dict:
        if self.verbose:
            print(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)

        input_data = [state[key] for key in input_keys]
        input_prompt = input_data[0]
        docs = input_data[1]
        ori_docs = [doc.page_content for doc in docs]
        format_docs = self._format_documents(docs)

        assert len(format_docs) > 0, "当前页面无法加载"

        output_parser = StrOutputParser()
        translate_result = []
        summary_result = []
        for index, doc in enumerate(ori_docs):
            translate_prompt = PromptTemplate(
                template="""
                        你是一个专业的文档翻译家，将我给出的Markdown文本，翻译成中文文档,在翻译的过程中注意下面的几个事项：
                        1. 不要修改文档的原有结构
                        2. 注意文档的专业词汇，不要强行翻译
                        3. 注意上下文的关系
                        以下是提供的文档：
                        <doc_provide>
                            {context}
                        </doc_provide>
                            {question}
                        """,
                input_variables=["question"],
                partial_variables={"context": doc},
            )

            translate_chain = translate_prompt | self.llm_model | output_parser
            answer = translate_chain.invoke({"question": input_prompt})
            translate_result.append(answer)

        for index, doc in enumerate(format_docs):
            summary_prompt = PromptTemplate(
                template="""
                            你是一个善于提炼文章结构的学习者，你的任务是对以下给出的文章进行分段总结。
                            首先，请阅读全文，然后按段落进行总结，每段总结不超过得少于一句话。
                            1. 请用简洁且富有创意的语言进行总结，但不要修改原文。
                            2. 如果有关键信息，可以进行标注。
                            3. 用中文进行返回
                            以下是提供的文档：
                            <doc_provide>
                                {context}
                            </doc_provide>
                                {question}
                                    """,
                input_variables=["question"],
                partial_variables={"context": doc},
            )

            summary_chain = summary_prompt | self.llm_model | output_parser
            answer = summary_chain.invoke({"question": input_prompt})
            summary_result.append(answer)

        new_docs = []
        for index, doc in enumerate(docs):
            doc.metadata["summary"] = summary_result[index]
            doc.metadata["translate"] = translate_result[index]
            new_docs.append(doc)

        state.update({self.output[0]: new_docs})
        return state

    def __init__(self, input: str, output: List[str],
                 node_config: Optional[dict] = None):
        super().__init__("summary", "node", input, output, min_input_len=1)
        self.verbose = node_config.get("verbose", False)
        self.llm_model = node_config["llm"]
