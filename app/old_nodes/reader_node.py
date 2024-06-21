from typing import List, Optional

from app.tools.loader.reader_loader import ReaderLoader
from app.old_nodes.base_node import BaseNode


class ReaderNode(BaseNode):

    def __init__(self, input: str, output: List[str], node_config: Optional[dict] = None, ):
        super().__init__(input=input, output=output, node_name="Reader",node_type="node", min_input_len=1)
        self.verbose = node_config.get("verbose", False)
        self.image_description = node_config.get("image_description", False)

    def execute(self, state: dict) -> dict:
        if self.verbose:
            self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)

        input_data = [state[key] for key in input_keys]
        source = input_data[0]

        docs = None
        if input_keys[0] == 'url':
            docs = ReaderLoader(
                query_or_url=source,
                image_description=self.image_description,
                mode='reader'
            ).load_and_split()

        elif input_keys[0] == 'query':
            docs = ReaderLoader(
                query_or_url=source,
                image_description=self.image_description,
                mode='search'
            ).load_and_split()

        state.update({self.output[0]: docs})
        return state
