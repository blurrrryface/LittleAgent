from typing import List, Optional

from app.old_nodes.base_node import BaseNode
from app.tools.models.openai_embeder import OpenaiEmbeder
from app.tools.storage.paradeDB import ParadeDB


class StorageNode(BaseNode):
    def __init__(self, input: str, output: List[str], node_config: Optional[dict] = None, ):
        super().__init__(input=input, output=output, node_name="storage", node_type="node", min_input_len=1)

        self.storage = ParadeDB(
            connection='postgresql+psycopg://pgvector:pgvector@localhost:5432/ai_dev',
            embedding_length=1536,
            embedding_function=OpenaiEmbeder()
        )
        self.verbose = node_config.get("verbose", False)
        self.storage_type = node_config.get("storage_type", "vector")


    def execute(self, state: dict) -> dict:
        """
        输入应该为2个参数  第一个值为存储的类型
        第二个参数为文章

        :param state:
        :return:
        """
        if self.verbose:
            self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)

        input_data = [state[key] for key in input_keys]
        docs = input_data[0]
        if self.storage_type == 'vector':
            if len(docs) == 0:
                return state
            self.storage.add_documents(docs)

        state.update({self.output[0]: docs})
        return state
