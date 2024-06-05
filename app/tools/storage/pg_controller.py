import sqlalchemy
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Text, String, JSON, TIMESTAMP, UniqueConstraint, \
    PrimaryKeyConstraint, text
from sqlalchemy.orm import sessionmaker
import os

# 加载 .env 文件
load_dotenv()
postgres_url = os.getenv('POSTGRES_HOST')
posgres_user = os.getenv('POSTGRES_USER')
postgres_password = os.getenv('POSTGRES_PASSWORD')
postgres_db = os.getenv('POSTGRES_DB')
postgres_port = os.getenv('POSTGRES_PORT')
postgres_url = f"postgresql+psycopg://{posgres_user}:{postgres_password}@{postgres_url}:{postgres_port}/{postgres_db}"

Base = sqlalchemy.orm.declarative_base()


class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='knowledge_base_pk'),
        UniqueConstraint('chunk_id', name='chunk_id_unique'),
        {'schema': 'document'}
    )

    id = Column(Integer, primary_key=True)
    document_content = Column(Text, nullable=True)
    url = Column(String, nullable=True)
    metadata_ = Column('metadata', JSON, nullable=True)  # 重命名属性以避免冲突
    source = Column(String, nullable=True)
    doc_id = Column(String, nullable=True)
    embedding = Column('embedding', String, nullable=True)  # Adjust type if necessary
    wordcount = Column(Integer, nullable=False)
    translation = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    chunk_id = Column(String, nullable=False)
    load_time = Column(TIMESTAMP, nullable=True, default='now()')
    links = Column(JSON, nullable=True)
    images = Column(JSON, nullable=True)


class KnowledgeBaseManager:
    def __init__(self):
        self.engine = create_engine(postgres_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def execute_query_all(self, sql):
        result = self.session.execute(text(sql))
        return result.fetchall()

    def execute_query_one(self, sql):
        result = self.session.execute(text(sql))
        return result.fetchone()._asdict()

    def update_by_id(self, record_id, **kwargs):
        self.session.query(KnowledgeBase).filter(KnowledgeBase.id == record_id).update(kwargs)
        self.session.commit()


# 使用示例
if __name__ == '__main__':

    manager = KnowledgeBaseManager()

    # 执行查询
    sql = 'SELECT * FROM document.knowledge_base;'
    sql2 = """
            SELECT count(1) num
            FROM document.knowledge_base 
            where translation is null;
            """
    sql3 = """
    SELECT id,document_content
    FROM document.knowledge_base 
    where translation is null;
    """
    results = manager.execute_query_all(sql3)
    iters = iter(results)
    print(next(iters))
    # for row in results:
    #     print(row)

    # 更新记录
    # manager.update_by_id(1, document_content='Updated content', url='http://newurl.com')
