import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import streamlit as st
import tempfile
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transcribe import transcribe_video

from rag import (
    chunk_text,
    create_vector_store,
    search_chunks
)


# PAGE CONFIGURATION


st.set_page_config(
    page_title="RAG AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)



# PROFESSIONAL UI DESIGN


st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #101827,
        #172554,
        #111827
    );
    color: white;
}

h1, h2, h3 {
    color: #60A5FA !important;
}

p, label {
    color: #E5E7EB;
}

.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #1D4ED8;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stChatMessage"] {
    background-color: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    padding: 15px;
}

.stTextInput input {
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    border: 1px solid #3B82F6;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)



# SIDEBAR


with st.sidebar:

    st.title("🎓 AI Teaching Assistant")

    st.write("Your personal AI learning companion.")

    st.divider()

    st.subheader("📌 Project Information")

    st.write("**Project:** RAG-Based AI Teaching Assistant")

    st.write("**Language:** Python")

    st.write("**AI Model:** FLAN-T5 Small")

    st.write("**Transcription:** Faster Whisper")

    st.write("**Vector Database:** FAISS")

    st.write("**Embeddings:** Sentence Transformers")

    st.write("**Interface:** Streamlit")

    st.divider()

    st.subheader("⚙️ How It Works")

    st.write("1. Upload educational video")

    st.write("2. Extract transcript")

    st.write("3. Create text chunks")

    st.write("4. Generate embeddings")

    st.write("5. Retrieve relevant information")

    st.write("6. Generate AI answers")

    st.divider()

    st.caption("Developed using Python and AI Technologies.")



# LOAD LANGUAGE MODEL


@st.cache_resource
def load_llm():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model



# GENERATE AI ANSWER


def generate_answer(question, context):

    tokenizer, model = load_llm()

    prompt = f"""
You are an intelligent educational teaching assistant.

Read the context carefully and answer the question.

Rules:
1. Understand what the question is asking.
2. Give a direct and complete answer.
3. Use only information available in the context.
4. Explain the answer in simple English.
5. Write 1 to 3 meaningful sentences.
6. Never give an incomplete phrase.
7. Do not copy the entire context.
8. If the context does not contain the answer,
   say "The answer is not available in the uploaded video."

Context:
{context}

Question:
{question}

Provide a complete answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            repetition_penalty=1.3,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return answer



# TIMESTAMP FORMAT


def get_video_timestamp(chunk):

    match = re.search(
        r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]",
        chunk
    )

    if not match:
        return "Timestamp unavailable"

    start = float(match.group(1))
    end = float(match.group(2))

    start_time = int(start)
    end_time = int(end)

    return (
        f"{start_time // 60:02d}:{start_time % 60:02d}"
        f" - "
        f"{end_time // 60:02d}:{end_time % 60:02d}"
    )



# SESSION STATE


if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []



# APPLICATION HEADER


st.title("🎓 RAG-Based AI Teaching Assistant")

st.write(
    "Upload an educational video and ask questions. "
    "Learn faster with AI-powered video understanding."
)

st.divider()


# VIDEO UPLOAD


st.header("📁 Upload Educational Video")

video = st.file_uploader(
    "Choose your educational video or audio file",
    type=["mp4", "mkv", "mov", "mp3", "wav"]
)


if video is not None:

    st.success(f"Uploaded successfully: {video.name}")

    if st.button("🚀 Process Video", type="primary"):

        suffix = os.path.splitext(video.name)[1]

        video_path = None

        st.session_state.transcript = ""
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.messages = []

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(video.getbuffer())

                video_path = temp_file.name


            # TRANSCRIPTION

            with st.spinner("🎧 Transcribing video using Whisper..."):

                transcript = transcribe_video(video_path)

            if not transcript or not transcript.strip():

                raise ValueError("No transcript generated.")

            st.session_state.transcript = transcript


            # TEXT CHUNKING

            with st.spinner("📚 Creating text chunks..."):

                chunks = chunk_text(transcript)

            if not chunks:

                raise ValueError("No text chunks generated.")

            st.session_state.chunks = chunks


            # VECTOR DATABASE

            with st.spinner("🧠 Creating FAISS vector database..."):

                index = create_vector_store(chunks)

            if index is None or index.ntotal == 0:

                raise ValueError("Vector database creation failed.")

            st.session_state.index = index

            st.success("🎉 Video processed successfully!")

            st.balloons()

        except Exception as e:

            st.error(f"Processing Error: {e}")

            st.session_state.transcript = ""
            st.session_state.chunks = []
            st.session_state.index = None

        finally:

            if video_path and os.path.exists(video_path):

                os.remove(video_path)



# DISPLAY PROCESSED VIDEO DATA


if st.session_state.transcript:

    st.divider()

    st.header("📊 Processing Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Chunks",
            len(st.session_state.chunks)
        )

    with col2:
        st.metric(
            "Total Vectors",
            st.session_state.index.ntotal
        )

    with col3:
        st.metric(
            "Status",
            "Completed"
        )


    # TRANSCRIPT

    st.header("📝 Video Transcript")

    with st.expander("View Full Transcript"):

        st.text_area(
            "Generated Transcript",
            st.session_state.transcript,
            height=200
        )


    # CHUNKS

    st.header("📚 Text Chunking")

    for i, chunk in enumerate(st.session_state.chunks):

        with st.expander(f"Chunk {i+1}"):

            st.write(chunk)


    # VECTOR DATABASE

    st.header("🗄️ FAISS Vector Database")

    st.success("Embeddings generated successfully!")

    st.write(
        "Total vectors stored:",
        st.session_state.index.ntotal
    )


    
    # CHATBOT
    
    st.divider()

    st.header("💬 AI Teaching Assistant Chatbot")

    st.write(
        "Ask any question related to your uploaded educational video."
    )


    # DISPLAY CHAT HISTORY

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                st.subheader("⏱️ Video References")

                for i, source in enumerate(message["sources"]):

                    timestamp = get_video_timestamp(
                        source["chunk"]
                    )

                    st.write(
                        f"Source {i+1}: {timestamp}"
                    )

                with st.expander("📚 View Retrieved Source Text"):

                    for i, source in enumerate(message["sources"]):

                        st.markdown(f"**Source {i+1}**")

                        st.write(source["chunk"])


    # CHAT INPUT

    question = st.chat_input(
        "Ask anything about your video..."
    )


    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)


        try:

            with st.chat_message("assistant"):

                # SEMANTIC SEARCH

                with st.spinner("🔍 Searching relevant information..."):

                    results = search_chunks(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        top_k=3
                    )


                if results:

                    context = "\n".join(
                        result["chunk"]
                        for result in results
                    )


                    # AI ANSWER

                    with st.spinner("🤖 Generating AI answer..."):

                        answer = generate_answer(
                            question,
                            context
                        )


                    if not answer:

                        answer = (
                            "Sorry, I could not generate an answer."
                        )

                    st.markdown(answer)


                    # VIDEO REFERENCES

                    st.subheader("⏱️ Video References")

                    for i, source in enumerate(results):

                        timestamp = get_video_timestamp(
                            source["chunk"]
                        )

                        st.write(
                            f"Source {i+1}: {timestamp}"
                        )


                    # RETRIEVED SOURCES

                    with st.expander("📚 View Retrieved Source Text"):

                        for i, source in enumerate(results):

                            st.markdown(f"**Source {i+1}**")

                            st.write(source["chunk"])


                    st.session_state.messages.append({

                        "role": "assistant",

                        "content": (
                            "Answer generated from retrieved context.\n\n"
                            + answer
                        ),

                        "sources": results

                    })


                else:

                    answer = "No relevant information found."

                    st.warning(answer)

                    st.session_state.messages.append({

                        "role": "assistant",

                        "content": answer,

                        "sources": []

                    })


        except Exception as e:

            st.error(f"Answer Generation Error: {e}")

            st.session_state.messages.append({

                "role": "assistant",

                "content": f"Error: {e}",

                "sources": []

            })


    # CLEAR CHAT

    if st.session_state.messages:

        if st.button("🗑️ Clear Chat"):

            st.session_state.messages = []

            st.rerun()




st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:15px;">

    <h4 style="color:#60A5FA;">
    🎓 RAG-Based AI Teaching Assistant
    </h4>

    <p>
    Python | Whisper | Sentence Transformers |
    FAISS | Hugging Face | Streamlit
    </p>

    <p>
    Intelligent Video Learning and Question Answering System
    </p>

    </div>
    """,
    unsafe_allow_html=True
)