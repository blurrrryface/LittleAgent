import enum
import hashlib
import json
import uuid
from typing import Type, List, Optional, Any, Iterable, Tuple
from sqlalchemy import create_engine, MetaData, text
import sqlalchemy

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VST
from sqlalchemy.orm import sessionmaker
from loguru import logger


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


class ParadeDB(VectorStore):
    def __init__(self,
                 connection: str
                 , engine_args=None
                 , embedding_length=None
                 , embedding_function: Optional[Embeddings] = None
                 , distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
                 ):

        self.tablename = 'document.knowledge_base'
        self._embedding_length = embedding_length
        self.distance_strategy = distance_strategy

        if isinstance(connection, str):
            self._engine = sqlalchemy.create_engine(
                url=connection, **(engine_args or {})
            )
        elif isinstance(connection, sqlalchemy.engine.Engine):
            self._engine = connection
        else:
            raise ValueError(
                "connection should be a connection string or an instance of "
                "sqlalchemy.engine.Engine"
            )
        self.session = sessionmaker(bind=self._engine)()
        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            raise ValueError("embedding_function is required")
        self.logger = logger

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def add_texts(self,
                  texts: Iterable[str],
                  metadatas: Optional[List[dict]] = None,
                  ids: Optional[List[str]] = None,
                  **kwargs: Any) -> List[str]:

        embeddings = self.embedding_function.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, docs_ids=ids, **kwargs
        )

    def add_documents(
            self,
            documents: List[Document],
            **kwargs: Any,
    ) -> List[str]:
        """当有documents时，先提取embeddings再插入
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        docs_id = str(uuid.uuid4())
        docs_ids = [docs_id] * len(texts)
        return self.add_texts(
            texts=texts, metadatas=metadatas, ids=docs_ids, **kwargs
        )

    @staticmethod
    def calculate_md5(text):
        # 创建一个MD5哈希对象
        md5_hash = hashlib.md5()

        # 更新哈希对象的值为文本
        md5_hash.update(text.encode('utf-8'))  # 必须将文本转换为字节，因此使用encode()

        # 获取哈希值
        hex_digest = md5_hash.hexdigest()

        return hex_digest

    def add_embeddings(
            self,
            texts: Iterable[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[dict]] = None,
            docs_ids: List[str] = None,  # docs_ids 是标识一整篇文章的标识
            **kwargs: Any,
    ) -> List[str]:
        """当有embeddings和文本时直接进行插入
        """

        if docs_ids is None:
            docs_ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        md5_for_text = [self.calculate_md5(text) for text in texts]

        connection = self._engine.connect()
        # Prepare data for the INSERT statement
        values = [
            (
                docs_id,
                metadata.get('url', ''),
                metadata.get('source', ''),
                str(embedding),
                document,
                json.dumps(metadata),
                len(document),
                chunk_id,
                json.dumps(metadata.get('links', {})),  # Ensure links is a JSON string
                json.dumps(metadata.get('images', {})),  # Ensure images is a JSON string
            )
            for document, metadata, embedding, docs_id, chunk_id in
            zip(texts, metadatas, embeddings, docs_ids, md5_for_text)
        ]
        query = text(f"""
                INSERT INTO {self.tablename} (document_content, url, metadata, embedding, source, doc_id,wordcount,chunk_id,links,images)
                VALUES (:document_content, :url, :metadata, :embedding, :source, :doc_id,:wordcount,:chunk_id,:links,:images)
                ON CONFLICT (chunk_id)
                DO UPDATE
                SET document_content = EXCLUDED.document_content,
                    url = EXCLUDED.url,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    source = EXCLUDED.source,
                    doc_id = EXCLUDED.doc_id,
                    wordcount = EXCLUDED.wordcount,
                    images = EXCLUDED.images,
                    links = EXCLUDED.links;
            """)

        # Execute the SQL statement
        with connection.begin():
            for value in values:
                self.logger.info(f"Inserting document : {value[2]}")
                connection.execute(query, {
                    'doc_id': value[0],
                    'url': value[1],
                    'source': value[2],
                    'embedding': value[3],
                    'document_content': value[4],
                    'metadata': value[5],
                    'wordcount': value[6],
                    'chunk_id': value[7],
                    'links': value[8],
                    'images': value[9],
                })
        return docs_ids

    @classmethod
    def __from(
            cls,
            texts: List[str],
            embeddings: List[List[float]],
            embedding_length: int,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
            connection: Optional[str] = None,
    ) -> VectorStore:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            connection=connection,
            embedding_length=embedding_length,
            distance_strategy=distance_strategy,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        return store

    @classmethod
    def from_texts(cls: Type[VST],
                   texts: List[str],
                   embedding: Embeddings,
                   metadatas: Optional[List[dict]] = None,
                   ids: Optional[List[str]] = None,
                   collection_name: Optional[str] = None,
                   distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
                   use_jsonb: bool = False,
                   **kwargs: Any) -> VST:
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            use_jsonb=use_jsonb,
            **kwargs,
        )

    def similarity_search(self
                          , query: str
                          , k: int = 4
                          , bm25_weight: float = 0.4
                          , similarity_weight: float = 0.6
                          , **kwargs: Any) -> List[Document]:
        sql_embedding = self.embedding_function.embed_query(query)
        sql_query = f"""
            select
                t1.document_content
                ,t1.doc_id
                ,t1.url
                ,t1.source
            from document.knowledge_base t1
            where t1.id in (
                SELECT id
                FROM knowledge_base_bm25.rank_hybrid(
                    bm25_query => 'document_content:"{query}"',
                    similarity_query => '''{sql_embedding}''<-> embedding',
                    bm25_weight => {bm25_weight},
                    similarity_weight => {similarity_weight}
                )
                limit {k}
            )
        """

        # 使用参数化查询
        result = self.session.execute(text(sql_query))
        documents = []
        for row in result:
            documents.append(
                Document(page_content=row.document_content,
                         metadata={'doc_id': row.doc_id, 'url': row.url, 'source': row.source}))
        return documents
