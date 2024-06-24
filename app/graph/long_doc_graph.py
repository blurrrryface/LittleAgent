from pprint import pprint
from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from loguru import logger
from pydantic import BaseModel,Field

from app.tools.models.select_llms import get_llms
from app.tools.storage.paradeDB import get_retriever_from_env


class LongDocGraphState(TypedDict):
    topic: str
    documents: List[Document]
    answers: List[str]
    generation: str


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class LongDocGraph:
    def __init__(self):
        '''
         第一步，根据主题，检索文章
         第二部，围绕一个主题，对于读取检索出来的文章，描写一个大纲
         第三部，将n个大纲拼接起来
         第四步，

        '''
        self.fast_llm = get_llms({
            "provider": "glm",
            "model": "GLM-4-0520",

        })
        self.llm = get_llms({
            "provider": "openai",
            "model": "gpt-4o-ca",

        })
        self.super_llm = get_llms({
            "provider": "openai",
            "model": "gpt-4-turbo-ca",

        })
        self.workflow = StateGraph(LongDocGraphState)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        self.workflow.add_node("answer", self.answer)
        self.workflow.add_node("summary", self.summary)
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_edge("grade_documents", "answer")
        self.workflow.add_edge("answer", "summary")
        # set up start and end nodes
        self.workflow.set_entry_point("retrieve")
        self.workflow.set_finish_point("summary")
        self.graph = self.workflow.compile()

    def retrieve(self, state):
        logger.info("正在检索文章中。。。")
        topic = state["topic"]

        db = get_retriever_from_env()
        retriever = db.as_retriever(
            search_kwargs={"k": 20}
        )

        # Retrieval
        documents = retriever.invoke(topic)
        return {"documents": documents, "topic": topic}

    def grade_documents(self, state):
        logger.info("正在判断文章是否与问题相关。。。")
        topic = state["topic"]
        documents = state["documents"]
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user topic. \n 
            If the document contains keyword(s) or semantic meaning related to the user topic, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the topic."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User topic: {topic}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"topic": topic, "document": d.metadata["summary"]}
            )
            grade = score["binary_score"]
            if grade == "yes":
                filtered_docs.append(d)
            else:
                continue
        return {"documents": filtered_docs, "topic": topic}

    def answer(self, state):
        logger.info("正在生成大纲。。。。")
        topic = state["topic"]
        documents = state["documents"]
        answers = []

        system = """是正在wiki问答编写者，现在你的任务是根据用户提供的Retrieved document，围绕用户提出的topic,
        回答用户的问题，你的问题不能超过文章提供的内容
        如果答案中包含链接 保留链接的内容"""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User topic: {topic}"),
            ]
        )

        answer_chain = answer_prompt | self.fast_llm | StrOutputParser()
        for doc in documents:
            answer = answer_chain.invoke(
                {"topic": topic, "document": doc.page_content}
            )
            answers.append(answer)
        return {"topic": topic, "documents": documents, "answers": answers}

    def summary(self, state):
        logger.info("正在生成答案")
        topic = state["topic"]
        answers = state["answers"]

        result_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "你是一个擅长知识梳理总结的学生，根据用户提供的相关答案集，综合所有的回答，撰写出一个完整的技术报告"),
                ("human", "用户提供的答案集: {answers} \n\n 用户所关系的主题: \n\n {topic}"),
            ]
        )
        result_chain = result_prompt | self.super_llm | StrOutputParser()

        result = result_chain.invoke(
            {"topic": topic, "answers": '====\nanswer:\n'.join(answers)}
        )
        return {"generation":result}


if __name__ == '__main__':
    graph = LongDocGraph().graph
    for output in graph.stream({"topic":"如何设计一个包含多个智能体的agent"}):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint(value["generation"])
