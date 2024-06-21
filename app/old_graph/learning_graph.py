from app.old_graph.abstract_graph import AbstractGraph
from app.old_graph.base_graph import BaseGraph
from app.old_nodes.reader_node import ReaderNode
from app.old_nodes.storage_node import StorageNode
from app.old_nodes.summary_node import SummaryNode


class LearningGraph(AbstractGraph):

    def __init__(self, config: dict, source: str):
        super().__init__(prompt='', config=config, source=source)
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
        return BaseGraph(
            nodes=[
                fetch_node,
                store_node,
                # summary_node
            ],
            edges=[
                (fetch_node, store_node),
                # (store_node, summary_node),
            ],
            entry_point=fetch_node
        )

    def run(self) -> str:
        inputs = {self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)
        docs = self.final_state.get("docs", None)
        if docs:
            return docs[0].page_content
        return "爬取失败"
