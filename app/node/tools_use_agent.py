import functools

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class ToolsUseAgent():
    def __init__(self, llm, tools, system_message: str,node_name: str):
        self.llm = llm
        self.tools = tools
        self.system_message = system_message
        self.node_name = node_name
        self.node = self.create_node()

    def create_node(self):
        agent = self.create_agent(
            self.llm,
            self.tools,
            system_message=self.system_message,
        )
        node = functools.partial(self.agent_node, agent=agent, name=self.node_name)
        return node

    def create_agent(self,llm, tools, system_message: str):
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

    # Helper function to create a node for a given agent
    def agent_node(self,state, agent, name):
        result = agent.invoke(state)
        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            "sender": name,
        }