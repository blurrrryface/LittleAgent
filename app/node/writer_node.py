from typing import List, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pydantic import BaseModel, Field

from app.node.structure_node import StructureNode


# class Subsection(BaseModel):
#     subsection_title: str = Field(..., title="文章小节的标题")
#     description: str = Field(..., title="文章小节的内容")
#
#     @property
#     def as_str(self) -> str:
#         return f"### {self.subsection_title}\n\n{self.description}".strip()
#
#
# class Section(BaseModel):
#     section_title: str = Field(..., title="章节的标题")
#     description: str = Field(..., title="章节的具体内容")
#     subsections: Optional[List[Subsection]] = Field(
#         default=None,
#         title="章节每个文章小节的标题和内容",
#     )
#
#     @property
#     def as_str(self) -> str:
#         subsections = "\n\n".join(
#             f"### {subsection.subsection_title}\n\n{subsection.description}"
#             for subsection in self.subsections or []
#         )
#         return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()
#
#
# class Outline(BaseModel):
#     page_title: str = Field(..., title="整篇文章的标题")
#     sections: List[Section] = Field(
#         default_factory=list,
#         title="文章每个章节的标题和内容",
#     )
#
#     @property
#     def as_str(self) -> str:
#         sections = "\n\n".join(section.as_str for section in self.sections)
#         return f"# {self.page_title}\n\n{sections}".strip()

class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()

class Writer(StructureNode):

    def __init__(self, llm_type: Literal["cheap", "fast", "cleaver"], **kwargs):
        super().__init__(llm_type, **kwargs)

    def get_agent(self):
        direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个wiki百科的作者. 根据用户提供的主题，撰写一个wiki界面.文章内容要全面且准确.",
                ),
                ("user", "{topic}"),
            ]
        )
        generate_outline_direct = direct_gen_outline_prompt | self.llm.with_structured_output(
            Outline
        )
        return generate_outline_direct


if __name__ == '__main__':
    writer = Writer('fast').get_agent()
    print(writer.invoke({"topic": "langGraph是什么"}))
