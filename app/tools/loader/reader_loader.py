import json
import logging
from typing import List, Any, Iterator, Dict, Optional
import re

import requests
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import urllib.parse

from langchain_text_splitters import TextSplitter, Language
from loguru import logger


class ReaderLoader(BaseLoader):
    """
    ReaderLoader 类用于从给定的查询或URL加载数据。

    :param query_or_url: 查询字符串或URL，用于数据的检索。
    :param image_description: 是否将图像描述作为搜索的特征，默认为False。
    :param mode: 操作模式，默认为'reader'。
    :param kwargs: 传递给其他方法的额外关键字参数。
    """

    def __init__(self,
                 query_or_url: str,
                 image_description: bool = False,
                 mode: str = 'reader',
                 **kwargs: Any):
        # 初始化超时时间、URLs、请求头和日志记录器
        self.timeout = 1000
        self.url = query_or_url
        self.mode = mode
        self.reader_url = 'https://r.jina.ai/'
        self.search_url = 'https://s.jina.ai/'
        # 根据是否是图像描述设置请求头
        self.headers = self._get_headers()

        # 获取日志记录器
        self.logger=logger


    def load(self) -> Optional[list[Document]]:
        """
        根据当前的模式（'reader' 或 'search'）从指定的 URL 加载文档内容。

        - 如果模式是 'reader'，则从 URL 加载 Markdown 格式的内容；
        - 如果模式是 'search'，则从 URL 加载用于搜索的内容；
        - 如果模式不是 'reader' 或 'search'，则记录错误并返回 None。

        返回值:
            - 如果加载成功，返回一个 Document 对象列表；
            - 如果加载失败或模式不支持，返回 None。
        """
        if self.mode == 'reader':
            docs = self._get_markdown_content(self.url)  # 加载 Markdown 内容
        elif self.mode == 'search':
            docs = self._get_search_content(self.url)  # 加载搜索内容
        else:
            self.logger.error(f'不支持的 mode：{self.mode}')  # 记录不支持的模式错误
            return None
        return docs


    async def aload(self) -> Optional[list[Document]]:
        if self.mode == 'reader':
            docs = await self._get_markdown_content(self.url)  # 加载 Markdown 内容
        elif self.mode == 'search':
            docs = await self._get_search_content(self.url)  # 加载搜索内容
        else:
            self.logger.error(f'不支持的 mode：{self.mode}')  # 记录不支持的模式错误
            return None
        return docs

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
                language=Language.MARKDOWN, chunk_size=1024*8, chunk_overlap=0
            )

        else:
            _text_splitter = text_splitter
        docs = self.load()
        return _text_splitter.split_documents(docs)

    def _validate_url(self, url: str) -> bool:
        # 这里只是一个简单的示例，实际情况可能需要更复杂的验证规则
        if not url.startswith('http') or not url.startswith('https'):
            return False
        return True

    def _get_headers(self ) -> dict[str, str]:

        return {
            'Accept': 'application/json',
            # 'X-With-Generated-Alt': str(image_description), # 老的接口
            "X-With-Links-Summary": "true",
            "X-With-Images-Summary": "true",
            "X-With-Generated-Alt": "true",

        }

    def _get_markdown_content(self, url: str) -> Optional[Document]:
        if not self._validate_url(url):
            self.logger.error(f'URL 验证失败：{url}')
            return None

        complete_url = f'{self.reader_url}{url}'
        self.logger.info(f'正在获取 {url} 的内容...')
        try:
            response = requests.get(complete_url, timeout=self.timeout, headers=self.headers)
        except requests.ConnectionError:
            self.logger.error(f'连接错误：{url}')
            return None
        except requests.Timeout:
            self.logger.error(f'请求超时：{url}')
            return None

        return self._handle_response(response)


    def _delete_images(self, markdown_text: str) -> str:
        """
        删除 Markdown 中的图片链接。保存在metadata中"""
        updated_text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'![\1]()', markdown_text)
        return updated_text

    def _handle_response(self, response: requests.Response) -> None | list[Document]:
        if response.status_code == 200:
            try:
                scrapy_content = json.loads(response.text)['data']
            except json.JSONDecodeError:
                self.logger.error(f'解析 JSON 失败：{response.text}')
                return None

            if isinstance(scrapy_content, dict):
                metadata = {
                    "source": scrapy_content['title'],
                    "url": scrapy_content['url'],
                    "images": scrapy_content['images'],
                    "links": scrapy_content['links']
                }

                return [Document(page_content=self._delete_images(scrapy_content['content']), metadata=metadata)]
            elif isinstance(scrapy_content, list):
                return [Document(page_content=self._delete_images(content['content']), metadata={
                    "source": content['title'],
                    "url": content['url'],
                    "images": content['images'],
                    "links": content['links']
                }) for content in scrapy_content]
            else:
                self.logger.error(f'未知的返回类型：{type(scrapy_content)} ,返回内容为{response.text}')
                return None
        else:
            self.logger.warning(f'访问失败，状态码：{response.status_code}，页面可能需要cookies才能访问，请检查')
            return None

    def _get_search_content(self, query: str) -> Optional[Document]:
        encode_query = urllib.parse.quote(query)
        complete_url = f'{self.search_url}{encode_query}'

        self.logger.info(f'正在查询 {query} 的内容...')
        try:
            response = requests.get(complete_url, timeout=self.timeout, headers=self.headers)
        except requests.ConnectionError:
            self.logger.error(f'连接错误')
            return None
        except requests.Timeout:
            self.logger.error(f'请求超时')
            return None

        return self._handle_response(response)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 设置日志记录器的级别为 INFO
    logger.setLevel(logging.INFO)

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 确保日志消息字符串完整
    logger.info('正在获取内容...')
