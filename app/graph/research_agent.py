from typing import TypedDict, List

from langgraph.checkpoint import MemorySaver
from langgraph.graph import StateGraph

from app.tools.models.select_llms import get_llms
import asyncio

class ResearchState(TypedDict):
    topic: str
    outline: Outline
    editors: List[Editor]
    interview_results: List[InterviewState]
    # The final sections output
    sections: List[WikiSection]
    article: str


class ResearchAgent:
    def __init__(self):
        self.llm = get_llms(llm_config)
        self.workflow = StateGraph(ResearchState)
        self.build_graph()
        self.graph = self.workflow.compile(checkpointer=MemorySaver())

    def build_graph(self):
        nodes = [
            ("init_research", initialize_research),
            ("conduct_interviews", conduct_interviews),
            ("refine_outline", refine_outline),
            ("index_references", index_references),
            ("write_sections", write_sections),
            ("write_article", write_article),
        ]
        for i in range(len(nodes)):
            name, node = nodes[i]
            self.workflow.add_node(name, node)
            if i > 0:
                self.workflow.add_edge(nodes[i - 1][0], name)
        self.workflow.set_entry_point(nodes[0][0])
        self.workflow.set_finish_point(nodes[-1][0])

    async def initialize_research(self, state):
        topic = state["topic"]
        coros = (
            generate_outline_direct.ainvoke({"topic": topic}),
            survey_subjects.ainvoke(topic),
        )
        results = await asyncio.gather(*coros)
        return {
            **state,
            "outline": results[0],
            "editors": results[1].editors,
        }


if __name__ == "__main__":
    pass
