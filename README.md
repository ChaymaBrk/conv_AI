# Conversational AI Platform

## Overview

Develop a FastAPI backend system with SQLAlchemy ORM for a conversational AI platform. The system should handle message processing, document management, and AI response generation using RAG for food queries and a weather API for weather queries

## Key Features

### Handling Messages

- **Classifying Messages:** When you ask a question, the system figures out whether it’s about **food** or **weather** and gives you an answer that fits.

- **Food Questions:**
  - If you’re asking for food advice, the platform uses a model called **Llama-3.3-70b**. This model helps the system pull the most relevant information from documents and give you an answer based on that.

- **Weather Questions:**
  - For weather-related queries, the platform gets the latest weather data, and then uses another model, **GPT-4o**, to explain the weather in a clear and simple way.

### Document Management

- **PDF Upload and Processing:**
  - You can upload PDF documents, and the system will automatically break them into pages, process them, and store them for future reference. This way, if you ask a question based on the document’s contents, it knows exactly where to look.

### Database Models

- **Message:** Keeps track of the messages that have been exchanged between you and the AI, including timestamps.
- **Document:** Manages the PDFs you upload, including tracking their status (whether they're still being processed or already finished).
- **DocumentPage:** Handles individual pages in a PDF to make sure everything is organized and searchable.

## Getting Started

### What You Need

Before you get started, make sure you have:

- **Python 3.10** or higher.
- A **virtual environment** (to keep things neat and tidy).
- API keys for the following services:
  - **OpenAI** (for GPT-4o and embeddings).
  - **Groq** (for Llama-3.3-70b).
  - A **Weather API** key (like from weatherapi.com).

### Steps to Install

1. **Clone the repository:**
   First, grab the code and move into the project directory:
   ```bash
   git clone <repository-url>
   cd conversational-AI
   ```

2. **Set up a virtual environment:**
   This keeps things organized:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   Now, just install everything you need:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Rename `.env.example` to `.env`.
   - Add your API keys inside the `.env` file like so:
     ```env
     OPENAI_API_KEY=your_openai_key
     GROQ_API_KEY=your_groq_key
     WEATHER_API_KEY=your_weather_key
     ```

5. **Set up the database:**
   Run this command to set everything up:
   ```bash
   python app/db.py
   ```

### Running the Application

1. **Start the server:**
   With the server up, run this to start it:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **API Docs:**
   Now you can go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to see the API docs and start testing out your queries.


## How It Works

### The Workflow

1. **Classifying Messages:** When you send a message, the system decides if it’s about food or weather. Then it processes the request:
   - **Food queries**: The system looks for relevant information from documents and gives you an answer.
   - **Weather queries**: It pulls the latest weather data and translates it into simple language.

2. **Managing Documents:** Upload a PDF, and the system breaks it down into smaller chunks. It stores these chunks in a database, so if you want to ask a question based on the document later, it can find the right part quickly.

3. **Database:** The system uses **SQLAlchemy** to handle all the data—messages, documents, and pages—so everything stays organized.

### Technologies We’re Using

- **FastAPI** for building the API (it’s fast and efficient!).
- **SQLAlchemy** for managing the database.
- **ChromaDB** for storing the document chunks and making them searchable.
- **OpenAI** for GPT-4o and embeddings (helps with generating text and understanding documents).
- **Groq** for Llama-3.3-70b (for food-related queries).
- **Weather API** for getting weather data.

```
