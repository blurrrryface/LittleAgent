from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.tools.models.select_llms import get_llms


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

if __name__ == '__main__':

    llm_config = {
        "provider":"openai",
        "model":"gpt-4o-ca"
    }
    llm = get_llms(llm_config)
    structured_llm_router = llm.with_structured_output(RouteQuery)
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
    print(
        question_router.invoke(
            {"question": "Who will the Bears draft first in the NFL draft?"}
        )
    )
    print(question_router.invoke({"question": "What are the types of node memory?"}))