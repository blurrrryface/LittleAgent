from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import FunctionMessage
from langchain.tools.render import format_tool_to_openai_function
from langgraph.graph import MessageGraph, END
import json

from app.tools.models.select_llms import get_llms

load_dotenv()

tools = [TavilySearchResults(max_results=3)]
tool_executor = ToolExecutor(tools)
model = get_llms({"provider": "openai",
    "model": "gpt-4o-ca"})
functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


def should_continue(messages):
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"


async def call_tool(messages):
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    response = await tool_executor.ainvoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return function_message


workflow = MessageGraph()


workflow.add_node("agent", model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")
app = workflow.compile()