import json
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.tools.models.select_llms import get_llms

load_dotenv()


def build_chain(llm, system_prompt, user_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ChatMessage(role="system", content=system_prompt + ",and write the response in chinese"),
        ChatMessage(role="user", content=user_prompt)
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


if __name__ == '__main__':
    llm_config_cleaver = {
        "provider": "openai",
        "model": "gpt-4o-ca",
        "response_format": {"type": "json_object"}
    }
    llm = get_llms(llm_config_cleaver)

    """
    你是一个论文撰写者，你的任务是从文献中摘抄出和主题相关的内容，保持文章的精简，但又不要丢失任何和主题有关的内容，并用中文进行回答
    """
