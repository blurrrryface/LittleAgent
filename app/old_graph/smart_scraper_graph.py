from app.old_graph.abstract_graph import AbstractGraph
from app.old_graph.base_graph import BaseGraph
from app.old_nodes.reader_node import ReaderNode
from app.old_nodes.storage_node import StorageNode
from app.old_nodes.summary_node import SummaryNode


class SmartScraperGraph(AbstractGraph):

    def __init__(self, prompt: str, config: dict, source: str):
        super().__init__(prompt, config, source)
        self.input_key = 'url'

    def _create_graph(self):
        fetch_node = ReaderNode(
            input="url",
            output=["docs"],
            node_config={
                "verbose": True,
            }
        )
        store_node = StorageNode(
            input="docs",
            output=["docs"],
            node_config={
                "storage_type": "vector",
                "verbose": True,
            }
        )
        summary_node = SummaryNode(
            input="prompt & docs",
            output=["summary"],
            node_config={
                "verbose": True,
                "llm": self.llm_model
            }
        )
        return BaseGraph(
            nodes=[
                fetch_node,
                store_node,
                summary_node
            ],
            edges=[
                (fetch_node, store_node),
                (store_node, summary_node),
            ],
            entry_point=fetch_node
        )

    def run(self) -> str:
        inputs = {"prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        return self.final_state.get("summary", "No answer found.")
