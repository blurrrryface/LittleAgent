from io import StringIO
from typing import Optional, List

import nbformat
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, Language
from loguru import logger
from nbconvert import MarkdownExporter
from streamlit.runtime.uploaded_file_manager import UploadedFile


class FileLoader(BaseLoader):
    def __init__(self, file: StringIO, file_type):
        self.file = file
        self.file_type = file_type

        self.logger = logger

    def convert_stringio_notebook_to_markdown(self, notebook_stringio):
        # 检查输入是否为StringIO对象
        if not isinstance(notebook_stringio, StringIO):
            raise TypeError("The input must be a StringIO object.")

        # 读取Notebook内容
        notebook_content = nbformat.read(notebook_stringio, as_version=4)

        # 创建Markdown导出器
        md_exporter = MarkdownExporter()

        # 将Notebook内容转换为Markdown
        (body, resources) = md_exporter.from_notebook_node(notebook_content)
        self.logger.info("ipynb 文件转化成功")

        return body

    def load(self)-> Optional[list[Document]]:
        if self.file_type == "ipynb":
            self.logger.info("开始读取ipynb文件")

            markdown_content = self.convert_stringio_notebook_to_markdown(self.file)
            # 创建一个Document对象
            document = Document(page_content=markdown_content, metadata={"source": self.file.name})

            return [document]
        elif self.file_type == "md":
            self.logger.info("开始读取md文件")
            # 创建一个Document对象
            document = Document(page_content=self.file.read(), metadata={"source": self.file.name})

            return [document]
        return None

    def load_and_split(
            self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        if text_splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError as e:
                raise ImportError(
                    "Unable to import from langchain_text_splitters. Please specify "
                    "text_splitter or install langchain_text_splitters with "
                    "`pip install -U langchain-text-splitters`."
                ) from e

            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN, chunk_size=1024 * 8, chunk_overlap=0
            )

        else:
            _text_splitter = text_splitter
        docs = self.load()
        return _text_splitter.split_documents(docs)