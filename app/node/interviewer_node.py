from typing import Annotated, List, Optional

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langsmith.run_helpers import as_runnable, traceable
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from app.node.editor_node import Editor
from app.node.structure_node import StructureNode
from app.tools.models.select_llms import get_llms


def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


def update_references(references, new_references):
    if not references:
        references = {}
    references.update(new_references)
    return references


def update_editor(editor, new_editor):
    # Can only set at the outset
    if not editor:
        return new_editor
    return editor


class InterviewState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], update_references]
    editor: Annotated[Optional[Editor], update_editor]


gen_qn_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced Wikipedia writer and want to edit a specific page. \
Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic. \
Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
Please only ask one question at a time and don't ask what you have asked before.\
Your questions should be related to the topic you want to write.
Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

Stay true to your specific perspective:

{persona}""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


def tag_with_name(ai_message: AIMessage, name: str):
    ai_message.name = name
    return ai_message


def swap_roles(state: InterviewState, name: str):
    converted = []
    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.dict(exclude={"type"}))
        converted.append(message)
    return {"messages": converted}


fast_llm = get_llms({
    "provider": "openai",
    "model": "gpt-4o-ca",
})


@as_runnable
@traceable
def generate_question(state: InterviewState):
    editor = state["editor"]
    gn_chain = (
            RunnableLambda(swap_roles).bind(name=editor.name)
            | gen_qn_prompt.partial(persona=editor.persona)
            | fast_llm
            | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = gn_chain.invoke(state)
    return {"messages": [result]}


if __name__ == '__main__':
    example_topic = '什么是langgraph'
    editor = Editor(
        affiliation="Open Source Community",
        name="CodeGenius",
        role="Machine Learning Enthusiast",
        description="CodeGenius will explore the open-source tools and frameworks available for implementing million-plus token context window language models in RAG systems and share their experiences with the community.",
    )
    messages = [
        HumanMessage(f"So you said you were writing an article on {example_topic}?")
    ]
    question = generate_question.invoke(
        {
            "editor": editor,
            "messages": messages,
        }
    )

    print(question["messages"][0].content)
