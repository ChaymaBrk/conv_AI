from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import openai
import PyPDF2
import chromadb
from uuid import uuid4
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()

# Database connection
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Message and Document models
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    is_ai = Column(Boolean, default=False)
    content = Column(String, index=True)
    timestamp = Column(String)

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    file_path = Column(String)
    is_processed = Column(Boolean, default=False)

class DocumentPage(Base):
    __tablename__ = "document_pages"
    
    document_id = Column(Integer, index=True)
    page_number = Column(Integer, primary_key=True)
    content = Column(String)
    is_processed = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

client = chromadb.Client()
collection = client.create_collection("documents")

def split_pdf_into_chunks(file_path, chunk_size=500):
    chunks = []
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Split text into chunks of `chunk_size` characters
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size].strip()
                    if chunk:
                        chunks.append({"page_number": page_num, "content": chunk})
    return chunks

def embed_chunks(chunks):
    
    texts = [chunk['content'] for chunk in chunks]
    response = openai.embeddings.create(
        model="text-embedding-3-small", 
        input=texts)
    
    return response.data[0].embedding
            
    
def store_chunks_in_chromadb(chunks, embeddings):
    for chunk, embedding in zip(chunks, embeddings):
        collection.add(
            documents=[chunk['content']],
            metadatas=[{"page_number": chunk['page_number']}],
            embeddings=[embedding],
            ids=[str(uuid4())]
        )


chunks =split_pdf_into_chunks(r"./food.pdf")
emb = embed_chunks(chunks)

store_chunks_in_chromadb(chunks,emb )