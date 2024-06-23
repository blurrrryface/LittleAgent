from typing import List, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langchain_core.pydantic_v1 import BaseModel, Field

from app.node.structure_node import StructureNode


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="文章小节的标题")
    description: str = Field(..., title="文章小节的内容")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="章节的标题")
    description: str = Field(..., title="章节的具体内容")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="章节每个文章小节的标题和内容",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="整篇文章的标题")
    sections: List[Section] = Field(
        default_factory=list,
        title="文章每个章节的标题和内容",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


class RelatedSubjects(BaseModel):
    topics: List[str] = Field(
        description="与主题相关的课题的调查列表.",
    )


class Writer(StructureNode):

    def __init__(self, llm_type, **kwargs):
        super().__init__(llm_type, **kwargs)

    def get_agent(self):
        direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个wiki写手.根据用户提供的主题撰写一个大纲. 大纲需要全面且明确.",
                ),
                ("user", "{topic}"),
            ]
        )
        generate_outline_direct = direct_gen_outline_prompt | self.llm.with_structured_output(
            Outline
        )
        return generate_outline_direct


class RelatedReacher(StructureNode):

    def __init__(self, llm_type, **kwargs):
        super().__init__(llm_type, **kwargs)

    def get_agent(self):
        gen_related_topics_prompt = ChatPromptTemplate.from_template(
            """我正在为下面提到的一个主题撰写维基百科页面。
            请确定并推荐一些与此主题密切相关的维基百科页面。
            我希望找到一些能够揭示与该主题常见相关方面有关的有趣内容，
            或者能够帮助我了解类似主题维基百科页面中通常包含的典型内容和结构的例子。
            请尽可能列出多个主题和网址。
    
            感兴趣的主题: {topic}"""
        )
        return gen_related_topics_prompt | self.llm.with_structured_output(
            RelatedSubjects
        )

if __name__ == '__main__':
    writer = RelatedReacher('fast').get_agent()
    print(writer.invoke({"topic": "langGraph是什么"}))
