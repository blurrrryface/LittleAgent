CREATE TABLE "document".knowledge_base (
	id int4 DEFAULT nextval('document.knowledge_base_doc_id_seq'::regclass) NOT NULL,
	document_content text NULL,
	url varchar NULL,
	metadata jsonb NULL,
	"source" varchar NULL,
	doc_id varchar NULL,
	embedding vector(1536) NULL,
	wordcount int4 NOT NULL,
	"translation" text NULL,
	summary text NULL,
	chunk_id varchar NOT NULL,
	load_time timestamp DEFAULT now() NULL,
	links jsonb NULL,
	images jsonb NULL,
	CONSTRAINT chunk_id_unique_1 UNIQUE (chunk_id),
	CONSTRAINT knowledge_base_pk_1 PRIMARY KEY (id)
);

CREATE INDEX knowledge_base_embedding_idx_1 ON document.knowledge_base
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CALL paradedb.create_bm25(
        index_name => 'knowledge_base_bm25',
        schema_name => 'document',
        table_name => 'knowledge_base',
        key_field => 'id',
        text_fields => '{
    translation: {tokenizer: {type: "chinese_lindera"}}, source: {tokenizer: {type: "icu"}}, summary: {tokenizer: {type: "chinese_lindera"}}
  }'
     );