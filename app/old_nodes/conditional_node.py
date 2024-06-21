from app.old_nodes.base_node import BaseNode


class ConditionalNode(BaseNode):
    def __init__(self, key_name: str, next_nodes: list, node_name="ConditionalNode"):
        super().__init__(node_name, "conditional_node")
        self.key_name = key_name
        if len(next_nodes) != 2:
            raise ValueError("next_nodes must contain exactly two elements.")
        self.next_nodes = next_nodes

    def execute(self, state: dict) -> dict:
        if self.key_name in state and len(state[self.key_name]) > 0:
            return self.next_nodes[0].node_name
        return self.next_nodes[1].node_name
