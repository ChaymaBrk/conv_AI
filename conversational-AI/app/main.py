from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import shutil
import time
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel
import psutil

# Define SQLAlchemy base and database session
Base = declarative_base()
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define the Message model
class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    is_ai = Column(Boolean, default=False)
    content = Column(String, index=True)
    timestamp = Column(String, default=lambda: datetime.utcnow().isoformat())

# Create tables
Base.metadata.create_all(bind=engine)

# Define the input schema for /messages endpoint
class MessageRequest(BaseModel):
    content: str

# FastAPI app setup
app = FastAPI(title="AI Assistant API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility function to handle file deletion errors
def remove_readonly(func, path, exc_info):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

def close_open_files(directory):
    """
    Ensure all file handles in the directory are closed.
    """
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for open_file in proc.open_files():
                if directory in open_file.path:
                    proc.terminate()  # Terminate the process using the file
        except Exception:
            pass  # Ignore errors from inaccessible processes

def cleanup_directory(directory):
    """
    Safely clean up a directory, ensuring no lingering file handles.
    """
    if os.path.exists(directory):
        # Wait briefly to allow processes to release locks
        time.sleep(0.5)

        # Ensure no processes are locking files
        close_open_files(directory)

        # Retry deleting the directory
        try:
            shutil.rmtree(directory, onerror=remove_readonly)
        except Exception as e:
            raise RuntimeError(f"Failed to delete {directory}: {str(e)}")

@app.post("/messages")
async def handle_message(content: MessageRequest, db: Session = Depends(get_db)):
    from services.classify_message import classify_query
    from services.rag_service import initialize_clients, store_document_in_chroma, generate_response
    from services.weather_service import get_weather_response

    classification = classify_query(content.content)

    # Initialize Chroma clients
    groq_client, docsearch = initialize_clients()

    try:
        pdf_path = "./food.pdf"
        store_document_in_chroma(docsearch, pdf_path)

    finally:
        # Clean up Chroma data
        cleanup_directory("./chroma_data")

    if classification == "food":
        response = generate_response(content.content)
    elif classification == "weather":
        response = get_weather_response()
    else:
        raise HTTPException(status_code=400, detail="Unsupported query type.")

    # Save messages to the database
    user_message = Message(is_ai=False, content=content.content)
    ai_message = Message(is_ai=True, content=response)
    db.add(user_message)
    db.add(ai_message)
    db.commit()

    return {"response": response}

@app.post("/documents")
async def process_document(file: UploadFile = File(...)):
    from services.process_documents import embed_chunks, store_chunks_in_chromadb, split_pdf_into_chunks

    try:
        # Save the uploaded file
        file_path = f"./uploads/{uuid4()}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Split PDF into chunks
        chunks = split_pdf_into_chunks(file_path)

        # Generate embeddings for the chunks
        embeddings = embed_chunks(chunks)

        # Store chunks and embeddings in ChromaDB
        store_chunks_in_chromadb(chunks, embeddings)

        # Return success response
        return {
            "message": "Document processed and stored successfully.",
            "num_chunks": len(chunks),
            "document_name": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Assistant API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)