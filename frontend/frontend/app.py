import os
from pathlib import Path
from typing import Any, Sequence

import streamlit as st
from openai import OpenAI
from sqlmodel import select
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import uuid

from database import open_session
from database.models import Chunk

# Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4-mini"
EMBEDDING_DIMENSIONS = 1536


def load_document(file_path: str | Path) -> Sequence[dict[str, Any]]:
    """Load a document using LangChain's UnstructuredFileLoader."""
    loader = UnstructuredFileLoader(str(file_path))
    return loader.load()


def create_chunks(
        documents: Sequence[dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 20
) -> Sequence[dict[str, Any]]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def store_document_chunks(
        chunks: Sequence[dict[str, Any]],
        title: str,
        chunk_size: int,
        chunk_overlap: int
) -> None:
    """Store document chunks in the database."""
    with open_session() as session:
        for i, chunk in enumerate(chunks):
            db_chunk = Chunk(
                id=uuid.uuid4(),
                chunk_content=chunk.page_content,
                chunk_title=title,
                page_number=i,
                meta={
                    **chunk.metadata,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
            session.add(db_chunk)
        session.commit()


def create_embedding(text: str, client: OpenAI) -> list[float]:
    """Create an embedding for a single text using OpenAI's API."""
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding


def process_pending_embeddings(client: OpenAI) -> int:
    """Create embeddings for all chunks that don't have them yet."""
    processed_count = 0

    with open_session() as session:
        chunks = session.exec(
            select(Chunk).where(Chunk.is_embedded == False)  # noqa: E712
        ).all()

        for chunk in chunks:
            chunk.embedding = create_embedding(chunk.chunk_content, client)
            chunk.is_embedded = True
            processed_count += 1

        session.commit()

    return processed_count


def find_relevant_chunks(
        query: str,
        client: OpenAI,
        k: int = 3
) -> Sequence[Chunk]:
    """Find the k most relevant chunks using vector similarity."""
    query_embedding = create_embedding(query, client)

    with open_session() as session:
        chunks = session.exec(
            select(Chunk)
            .where(Chunk.is_embedded == True)  # noqa: E712
            .order_by(Chunk.embedding.l2_distance(query_embedding))
            .limit(k)
        ).all()

        return chunks


def generate_answer(
        query: str,
        client: OpenAI,
        k: int = 3
) -> str:
    """Generate an answer for a query using relevant chunks and GPT-4."""
    relevant_chunks = find_relevant_chunks(query, client, k)

    if not relevant_chunks:
        return "No relevant information found in the documents."

    # Prepare context from chunks
    context = "\n\n".join(
        f"[From {chunk.chunk_title}, Page {chunk.page_number + 1}]:\n{chunk.chunk_content}"
        for chunk in relevant_chunks
    )

    # Create prompt
    prompt = f"""Based on the following excerpts from documents, please answer the question.
    If the answer cannot be found in the excerpts, say "I cannot find relevant information to answer this question."

    Document Excerpts:
    {context}

    Question: {query}

    Answer: """

    # Get GPT-4 response
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


def calculate_embedding_cost(texts: Sequence[dict[str, Any]]) -> tuple[int, float]:
    """Calculate the cost of creating embeddings."""
    enc = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
    total_tokens = sum(len(enc.encode(text.page_content)) for text in texts)
    # Current pricing for text-embedding-3-small is $0.0004 per 1K tokens
    cost = (total_tokens / 1000) * 0.0004
    return total_tokens, cost


def process_documents(
        uploaded_files: Sequence[Any],
        chunk_size: int,
        chunk_overlap: int,
        client: OpenAI
) -> None:
    """Process uploaded documents and store them in the database."""
    temp_dir = Path('./temp_docs')
    temp_dir.mkdir(exist_ok=True)

    try:
        all_chunks = []
        for uploaded_file in uploaded_files:
            # Save file temporarily
            temp_path = temp_dir / uploaded_file.name
            temp_path.write_bytes(uploaded_file.read())

            # Process document
            documents = load_document(temp_path)
            chunks = create_chunks(documents, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)

            # Store in database
            store_document_chunks(chunks, uploaded_file.name, chunk_size, chunk_overlap)

            st.sidebar.write(f'Processed: {uploaded_file.name} ({len(chunks)} chunks)')

        # Calculate and display embedding cost
        tokens, cost = calculate_embedding_cost(all_chunks)
        st.sidebar.write(f'Embedding cost: ${cost:.4f} ({tokens:,} tokens)')

        # Create embeddings
        with st.sidebar.spinner('Creating embeddings...'):
            processed_count = process_pending_embeddings(client)
            st.sidebar.success(f'Created {processed_count} embeddings')

    finally:
        # Cleanup temporary files
        for file in temp_dir.iterdir():
            file.unlink()
        temp_dir.rmdir()


def main() -> None:
    """Main application entry point."""
    # Initialize Streamlit page
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide"
    )
    st.header("Document Question-Answering System")

    # Initialize session state
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ''

    # Setup sidebar
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            client = OpenAI()
        else:
            client = None

        uploaded_files = st.file_uploader(
            'Upload documents to analyze:',
            accept_multiple_files=True
        )

        chunk_size = st.number_input(
            'Chunk size:',
            min_value=100,
            max_value=8192,
            value=512,
            help="Number of characters per chunk. Larger chunks provide more context but cost more to embed."
        )

        chunk_overlap = st.number_input(
            'Chunk overlap:',
            min_value=0,
            max_value=chunk_size // 2,
            value=20,
            help="Number of characters that overlap between chunks to maintain context."
        )

        k = st.number_input(
            'Number of relevant chunks:',
            min_value=1,
            max_value=20,
            value=3,
            help="Number of most relevant chunks to use when answering questions."
        )

    # Handle document processing
    if uploaded_files and st.sidebar.button('Process Documents'):
        if not client:
            st.sidebar.error('Please provide your OpenAI API key.')
            st.stop()

        with st.spinner('Processing documents...'):
            process_documents(uploaded_files, chunk_size, chunk_overlap, client)

    # Setup query interface
    query = st.text_input('Ask a question about your documents:', key='text_input')

    if query:
        if not client:
            st.error('Please provide your OpenAI API key to ask questions.')
            st.stop()

        with st.spinner('Finding answer...'):
            answer = generate_answer(query, client, k)
            st.text_area('Answer:', value=answer, height=200)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('New Question'):
                st.session_state.text_input = ''
                st.experimental_rerun()
    else:
        st.info('Upload documents and ask questions to get started.')

    # Add system information
    st.markdown('---')
    st.subheader('About this System')
    st.markdown("""
    This document Q&A system uses advanced natural language processing to help you interact with your documents:

    1. Documents are split into manageable chunks and stored in a PostgreSQL database
    2. Each chunk is embedded using OpenAI's text-embedding-3-small model
    3. Questions are matched to relevant document chunks using vector similarity
    4. GPT-4 generates answers based on the most relevant chunks

    The system uses pgvector for efficient similarity search and can handle multiple document formats.
    """)


if __name__ == "__main__":
    main()