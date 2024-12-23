from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from services.classify_message import classify_query
from services.weather_service import get_weather_response
from models import Message
from fastapi import APIRouter, UploadFile, File
import os
router = APIRouter()
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from uuid import uuid4
from services.process_documents import embed_chunks, store_chunks_in_chromadb,split_pdf_into_chunks
from services.rag_service import initialize_clients, store_document_in_chroma, generate_response

@router.post("/messages")
 
async def handle_message( content, db: Session = Depends(get_db)):
    
    classification = classify_query(content)
    
    groq_client, docsearch = initialize_clients()
    
    try:
        pdf_path = "./food.pdf"
        store_document_in_chroma(docsearch, pdf_path)

    
    finally:
        # Clean up Chroma data
        if os.path.exists("./chroma_data"):
            shutil.rmtree("./chroma_data")

    if classification == "food":
        response = generate_response(content)
    elif classification == "weather":
        response = get_weather_response()
    else:
        return {"error": "Unsupported query type."}

    # Save messages to the database
    user_message = Message(is_ai=False, content=content)
    ai_message = Message(is_ai=True, content=response)
    db.add(user_message)
    db.add(ai_message)
    db.commit()

    return {"response": response}


# Endpoint to process a PDF document
@router.post("/documents")
async def process_document(file: UploadFile = File(...)):
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

