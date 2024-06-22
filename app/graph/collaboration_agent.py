from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.node.tools_use_agent import ToolsUseAgent
from app.tools.models.select_llms import get_llms
from app.tools.normal_tools import tavily_tool, python_repl

load_dotenv()

import operator
from typing import Annotated, Sequence, TypedDict, Literal


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


@traceable  # Auto-trace this function
class CollaborationAgent:
    def __init__(self):
        llm_config = {
            "provider": "openai",
            "model": "gpt-4o-ca",
        }
        self.llm = get_llms(llm_config)
        tools = [tavily_tool, python_repl]
        tool_node = ToolNode(tools)
        research_node = ToolsUseAgent(
            self.llm,
            [tavily_tool],
            ("You should provide accurate data for use, "
             "and source code shouldn't be the final answer,only when you success run the code,and get the result,you can output the final answer"),
            "Researcher"
        ).node
        chart_node = ToolsUseAgent(
            self.llm,
            [python_repl],
            "Run the python code to display the chart.",
            "chart_generator"
        ).node
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("Researcher", research_node)
        self.workflow.add_node("chart_generator", chart_node)
        self.workflow.add_node("call_tool", tool_node)

        self.workflow.add_conditional_edges(
            "Researcher",
            self.router,
            {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
        )
        self.workflow.add_conditional_edges(
            "chart_generator",
            self.router,
            {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
        )

        self.workflow.add_conditional_edges(
            "call_tool",
            # Each agent node updates the 'sender' field
            # the tool calling node does not, meaning
            # this edge will route back to the original agent
            # who invoked the tool
            lambda x: x["sender"],
            {
                "Researcher": "Researcher",
                "chart_generator": "chart_generator",
            },
        )
        self.workflow.set_entry_point("Researcher")
        self.graph = self.workflow.compile()

    def router(self, state) -> Literal["call_tool", "__end__", "continue"]:
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            # The previous agent is invoking a tool
            return "call_tool"
        if "FINAL ANSWER" in last_message.content:
            # Any agent decided the work is done
            return "__end__"
        return "continue"


if __name__ == '__main__':
    graph = CollaborationAgent().graph
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Fetch the UK's GDP over the past 5 years,"
                            " then draw a line graph of it."
                            " Once you code it up, finish."
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )
    for s in events:
        print(s)
        print("----")
