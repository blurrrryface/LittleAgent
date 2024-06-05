from app.nodes.base_node import BaseNode


class DBFetchNode(BaseNode):
    def __init__(self):
        super().__init__(

            node_name="DBFetchNode",
        )

    def execute(self, state: dict) -> dict:

        return state