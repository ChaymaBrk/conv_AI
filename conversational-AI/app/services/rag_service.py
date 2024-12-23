from chromadb import Client as ChromaClient
from chromadb.config import Settings
import os
from groq import Groq
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from uuid import uuid4
from dotenv import load_dotenv
import shutil
import chromadb
load_dotenv()


from langchain_openai import OpenAIEmbeddings  # Note the import from langchain_openai

def initialize_clients():
    # Initialize Groq
    groq_api_key = os.getenv('GROQ_API_KEY')
    groq_client = Groq(api_key=groq_api_key)
    
    # Initialize OpenAI embeddings
    openai_api_key = os.getenv('OPENAI_API_KEY')  # Make sure to set this in your environment
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    chroma_dir = "./chroma_data"
    try:
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
    except PermissionError:
        print("Warning: Could not delete existing database")
        
    os.makedirs(chroma_dir, exist_ok=True)
    
    # Use OpenAI embeddings with Chroma
    docsearch = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )
    
    return groq_client, docsearch


# def store_document_in_chroma(docsearch, pdf_path):
#     try:
#         # Your existing document processing code here
#         docsearch.add_texts([chunk], ids=[doc_id])
#     finally:
#         # Ensure the database is properly persisted
#         if hasattr(docsearch, '_client'):
#             docsearch._client.persist()

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text_chunks = [page.extract_text() for page in reader.pages if page.extract_text()]
        return text_chunks
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return []

def store_document_in_chroma(docsearch, pdf_path):
    text_chunks = extract_text_from_pdf(pdf_path)
    if not text_chunks:
        print("No text extracted from PDF.")
        return
    
    for chunk in text_chunks:
        doc_id = str(uuid4())
        docsearch.add_texts([chunk], ids=[doc_id])
    print("Document successfully stored in Chroma.")

def get_relevant_excerpts(docsearch, user_question):
    try:
        relevant_docs = docsearch.similarity_search(user_question)
        excerpts = '\n\n------------------------------------------------------\n\n'.join(
            [doc.page_content for doc in relevant_docs[:3]]
        )
        return excerpts
    except Exception as e:
        print(f"Error retrieving relevant excerpts: {str(e)}")
        return ""

def generate_response(groq_client, user_question, relevant_excerpts):
    system_prompt = '''
    You are an expert assistant. Based on the user's question and relevant excerpts from the documents,
    provide an accurate response. Include references to the excerpts wherever applicable.
    '''
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"User Question: {user_question}\n\nRelevant Excerpts:\n\n{relevant_excerpts}"
                }
            ],
            model="llama-3.3-70b-versatile"
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Unable to generate a response at this time."

def main():
    print("PDF Document QA using Chroma and Groq Llama")
    
    # Initialize clients
    groq_client, docsearch = initialize_clients()
    
    try:
        pdf_path = "./food.pdf"
        store_document_in_chroma(docsearch, pdf_path)
        
        while True:
            user_question = input("Enter your question (or 'quit' to exit): ")
            if user_question.lower() == 'quit':
                break
                
            relevant_excerpts = get_relevant_excerpts(docsearch, user_question)
            if relevant_excerpts:
                response = generate_response(groq_client, user_question, relevant_excerpts)
                print(f"\nResponse:\n{response}\n")
            else:
                print("No relevant excerpts found.")
    
    finally:
        # Clean up Chroma data
        if os.path.exists("./chroma_data"):
            shutil.rmtree("./chroma_data")

if __name__ == "__main__":
    main()