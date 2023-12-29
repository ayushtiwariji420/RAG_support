import os
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores import SupabaseVectorStore
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores import Pinecone
from supabase.client import Client, create_client
import pinecone


class Vectorizer:
    def __init__(self, embeddings=OpenAIEmbeddings()):
        self.embeddings = embeddings

    def ready_csv_content(self, csv_path, content_column:str=None):

        df = pd.read_csv(csv_path)
        docs = []

        for index, row in df.iterrows():
            query_text = row[content_column]
            metadata = {col: row[col] for col in df.columns if col != content_column}
            document = Document(
                page_content=query_text,
                metadata=metadata,
            )
            docs.append(document)
        
        return docs
    
    def read_txt_content(self,txt_path):
        loader = TextLoader(txt_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs

    def ready_pdf_content(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(documents)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs

class DBFaiss(Vectorizer):
    def __init__(self, embeddings=OpenAIEmbeddings()):
        super().__init__(embeddings)

    def create_db_faiss(self, docs:list, name):
        db = FAISS.from_documents(docs, self.embeddings)
        db.save_local(name)
    
    def query_faiss(self, name, query):
        new_db = FAISS.load_local(name, self.embeddings)
        docs = new_db.similarity_search(query)
        return docs[0]

class DBSupabase(Vectorizer):
    def __init__(self,supabase_url, supabase_key,  embeddings=OpenAIEmbeddings()):
        super().__init__(embeddings)
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

    def create_db_supabase(self, docs:list,table_name="documents", query_name="match_documents"):
        SupabaseVectorStore.from_documents(
            docs,
            self.embeddings,
            client=self.supabase,
            table_name=table_name,
            query_name=query_name,
            chunk_size=500,
        )
    
    def query_supabase(self,query, table_name="documents", query_name="match_documents"):
        vector_store = SupabaseVectorStore(
            embedding=self.embeddings,
            client=self.supabase,
            table_name=table_name,
            query_name=query_name
        )
        matched_docs = vector_store.similarity_search(query)
        return matched_docs[0]

class DBPgvector(Vectorizer):
    def __init__(self, username:str, host:str, database:str, port:int=5432,  embeddings=OpenAIEmbeddings()):
        super().__init__(embeddings)
        self.CONNECTION_STRING = f"postgresql+psycopg2://{username}@{host}:{port}/{database}"

    def create_db_pgvector(self,docs:list, collection_name:str="default_connection"):
        PGVector.from_documents(
            embedding=self.embeddings,
            documents=docs,
            collection_name=collection_name,
            connection_string=self.CONNECTION_STRING
        )
    
    def query_pgvector():
        pass

class DBPinecone(Vectorizer):

    def __init__(self, pinecone_api_key, pinecone_env, embeddings=OpenAIEmbeddings()):
        super.__init__(embeddings)
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        pinecone.init(
            api_key= self.pinecone_api_key,
            environment= self.pinecone_env
        )

    def create_db_pinecone(self, docs:list, index_name:str):
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

        Pinecone.from_documents(docs, self.embeddings, index_name=index_name)

    def query_pinecone(self, query:str, index_name:str):
        docsearch = Pinecone(self.embeddings, index_name=index_name)
        docs = docsearch.similarity_search(query)
        return docs[0]