from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from app.node.structure_node import StructureNode


class Editor(BaseModel):
    affiliation: str = Field(
        description="编辑的主要从属关系.",
    )
    name: str = Field(
        description="编辑的名字.", pattern=r"^[a-zA-Z0-9_-]{1,64}$"
    )
    role: str = Field(
        description="在这个话题的背景下，编辑的角色是什么.",
    )
    description: str = Field(
        description="编辑的关注点、考虑和动机的描述.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="编辑人员的全面列表及其角色和隶属关系.",
        # Add a pydantic validation/restriction to be at most M editors
        min_items=8,

    )


class EditorNode(StructureNode):
    def __init__(self, llm_type, **kwargs):
        super().__init__(llm_type, **kwargs)

    def get_agent(self):
        gen_perspectives_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """您需要选择一个多样化（且独特）的维基百科编辑团队，
                    他们将合作撰写一篇关于该主题的全面文章。
                    每个编辑应该带来与这个主题相关的不同观点、角色或从属关系。
                    您可以参考其他类似主题的维基百科页面以获得灵感。
                    为每个编辑提供描述，概述他们各自的专注领域:
            {examples}""",
                ),
                ("user", "感兴趣的主题: {topic}"),
            ]
        )
        gen_perspectives_chain = gen_perspectives_prompt | self.llm.with_structured_output(
            Perspectives
        )
        return gen_perspectives_chain

if __name__ == '__main__':
    writer = EditorNode('fast').get_agent()
    print(writer.invoke({"topic": "langGraph是什么"}))
