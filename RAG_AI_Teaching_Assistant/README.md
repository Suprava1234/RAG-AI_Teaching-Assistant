🎓 RAG-Based AI Teaching Assistant

Project Overview

The RAG-Based AI Teaching Assistant is an educational application that allows users to upload educational videos and ask questions about their content.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from video transcripts and generate answers using an AI language model.

Features

- Upload educational video and audio files.
- Automatic speech transcription using Whisper.
- Text chunking for efficient information retrieval.
- Text embeddings using Sentence Transformers.
- FAISS vector database for semantic search.
- AI-powered question answering using FLAN-T5.
- Answer verification with context-based fallback.
- Video timestamp references.
- Interactive chatbot using Streamlit.
- Chat history and clear chat functionality.

Technologies Used

Technology| Purpose
Python| Programming language
Streamlit| User interface
Faster-Whisper| Video transcription
Sentence Transformers| Text embeddings
FAISS| Vector database
Hugging Face| Language model
FLAN-T5 Small| Answer generation
NumPy| Numerical operations

Project Workflow

1. Upload an educational video.
2. Extract spoken content using Whisper.
3. Divide transcript into smaller text chunks.
4. Generate embeddings using Sentence Transformers.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks using semantic search.
7. Generate answers using FLAN-T5.
8. Verify answers against retrieved context.
9. Display answers and video timestamps.

Project Structure

RAG_AI_Teaching_Assistant/
│
├── app.py
├── rag.py
├── transcribe.py
├── requirements.txt
└── README.md

Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project folder:

cd RAG_AI_Teaching_Assistant

Install dependencies:

pip install -r requirements.txt

Run the Application

streamlit run app.py

Open the local Streamlit URL displayed in the terminal.

Testing

Example questions:

- What is photosynthesis?
- What is the role of chlorophyll?
- What are the raw materials required for photosynthesis?
- What are the products of photosynthesis?

Future Improvements

- Improve AI answer accuracy.
- Add support for multiple educational videos.
- Add clickable video timestamps.
- Improve chatbot interface.
- Support multilingual video transcription.

Conclusion

This project demonstrates how Retrieval-Augmented Generation can be used to build an educational AI assistant that helps students understand video content through semantic search and AI-powered question answering.

Developed using Python, Whisper, FAISS, Hugging Face and Streamlit.