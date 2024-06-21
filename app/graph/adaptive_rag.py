import os
from pprint import pprint
from typing import List, Literal

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.tools.models.select_llms import get_llms
from app.tools.storage.paradeDB import get_retriever_from_env

load_dotenv()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class AdaptiveRag:
    def __init__(self, llm):
        self.workflow = StateGraph(GraphState)
        self.llm = llm
        self.workflow.add_node("retrieve", self.retrieve)  # retrieve
        self.workflow.add_node("web_search", self.web_search)  # retrieve
        self.workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.generate)  # generatae
        self.workflow.add_node("transform_query", self.transform_query)  # transform_query
        self.__build_graph()
        self.app = self.workflow.compile()

    def grade_documents(self, state):
        logger.info("判断文章是否与问题相关")
        question = state["question"]
        documents = state["documents"]
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                logger.info("文章相关性判断结果：相关")
                filtered_docs.append(d)
            else:
                logger.info("文章相关性判断结果：不相关")
                continue
        return {"documents": filtered_docs, "question": question}

    def generate(self, state):
        print("尝试回答问题中")
        question = state["question"]
        documents = state["documents"]

        template = """使用下面的检索到的文章回答我的问题

                        {context}

                        Question: {question}

                        你的回答:"""
        custom_rag_prompt = PromptTemplate.from_template(template)
        rag_chain = (
                custom_rag_prompt
                | self.llm
                | StrOutputParser()
        )

        # RAG generation
        generation = rag_chain.invoke({"context": self.format_docs(documents), "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def retrieve(self, state):
        logger.info("检索文章中")
        question = state["question"]
        documents = state["documents"]

        db = get_retriever_from_env()
        retriever = db.as_retriever(
            search_kwargs={"k": 4}
        )

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def web_search(self, state):
        logger.info("搜索互联网资源中")
        question = state["question"]
        web_search_tool = TavilySearchResults(k=3)
        # Web search
        docs = web_search_tool.invoke({"query": question})
        # print(docs)
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    def transform_query(self, state):
        logger.info("问题重写中")
        question = state["question"]
        documents = state["documents"]
        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
             for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        question_rewriter = (
                re_write_prompt
                | self.llm
                | StrOutputParser()
        )

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def route_question(self, state):

        logger.info("选择检索方式")
        question = state["question"]
        structured_llm_router = self.llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | structured_llm_router
        source = question_router.invoke({"question": question})
        # print(source)
        if source["datasource"] == "web_search":
            logger.info("选择网页检索")
            return "web_search"
        elif source["datasource"] == "vectorstore":
            logger.info("选择文章检索")
            return "vectorstore"

    def decide_to_generate(self, state):
        logger.info("对检索出来的文章进行评级中")
        # state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:

            logger.info("所有文章都和检索内容不相关，开始重写问题")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            logger.info("开始回答问题")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        logger.info("检查问题可信度中")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)

        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_grader = hallucination_prompt | structured_llm_grader

        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
             Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | structured_llm_grader

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["binary_score"]

        # Check hallucination
        if grade == "yes":
            logger.info("生成的内容是可信的")
            # Check question-answering
            logger.info("检查生成的内容是否回答了问题")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score["binary_score"]
            if grade == "yes":
                logger.info("生成的内容回答了问题")
                return "useful"
            else:
                logger.info("生成的内容没有回答问题")
                return "not useful"
        else:
            logger.info("生成的内容是不可信的")
            return "not supported"

    def __build_graph(self):
        # Build graph
        self.workflow.set_conditional_entry_point(
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

    def format_docs(self, docs):
        # print(docs)
        # print(type(docs))
        if isinstance(docs,Document):
            return docs.page_content
        else:
            return "\n\n".join(doc.page_content for doc in docs)


if __name__ == '__main__':
    # Run
    inputs = {
        "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
    }
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o-ca"
    }
    llm = get_llms(llm_config)
    app = AdaptiveRag(llm).app
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])
