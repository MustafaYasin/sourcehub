import os
from pathlib import Path
from typing import Any, Sequence

import streamlit as st
from openai import AsyncOpenAI, OpenAI
from sqlmodel import select
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import uuid
from sqlalchemy import func

from database import open_session
from database.models import Chunk

# Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4"
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


def create_embeddings(client: OpenAI, chunks: Sequence[Any]) -> int:
    """Create embeddings for multiple chunks in a single batch."""
    if not chunks:
        return 0

    try:
        # Get all chunk contents in a list - use page_content for LangChain documents
        texts = [chunk.page_content for chunk in chunks]

        # Create embeddings for all texts in one API call
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
            encoding_format="float"
        )

        # Now store in database as Chunk objects
        with open_session() as session:
            for i, (doc, embedding_data) in enumerate(zip(chunks, response.data)):
                db_chunk = Chunk(
                    id=uuid.uuid4(),
                    chunk_content=doc.page_content,
                    chunk_title=doc.metadata.get('source', 'Unknown'),
                    page_number=i,
                    embedding=embedding_data.embedding,
                    is_embedded=True,
                    meta=doc.metadata
                )
                session.add(db_chunk)
            session.commit()

        return len(chunks)

    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return 0


def process_pending_embeddings(client: OpenAI, batch_size: int = 100) -> int:
    """Process all pending embeddings in batches."""
    total_processed = 0

    with open_session() as session:
        # Get all unembedded chunks
        chunks = session.exec(
            select(Chunk).where(Chunk.is_embedded == False)  # noqa: E712
        ).all()

        if not chunks:
            return 0

        # Process in batches
        total_chunks = len(chunks)
        progress_bar = st.progress(0)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            try:
                # Get texts for the batch
                texts = [chunk.chunk_content for chunk in batch]

                # Create embeddings
                response = client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=texts,
                    encoding_format="float"
                )

                # Update chunks with embeddings
                for chunk, embedding_data in zip(batch, response.data):
                    chunk.embedding = embedding_data.embedding
                    chunk.is_embedded = True

                # Commit the batch
                session.commit()
                total_processed += len(batch)

                # Update progress
                progress = min((i + batch_size) / total_chunks, 1.0)
                progress_bar.progress(progress)
                st.write(f"Processed {min(i + batch_size, total_chunks)} of {total_chunks} chunks")

            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")
                session.rollback()
                break

        progress_bar.empty()

    return total_processed


def create_embedding(text: str, client: OpenAI) -> list[float]:
    """Create a single embedding using OpenAI's API."""
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding


def clean_database() -> None:
    """Clean database by removing duplicate entries based on content and title."""
    with open_session() as session:
        # First, let's get all chunks
        all_chunks = session.exec(select(Chunk)).all()

        # Create a set to track unique content
        seen_content = set()
        chunks_to_delete = []

        for chunk in all_chunks:
            # Create a unique identifier for each chunk based on content and title
            content_key = (chunk.chunk_content, chunk.chunk_title)

            if content_key in seen_content:
                chunks_to_delete.append(chunk)
            else:
                seen_content.add(content_key)

        # Delete duplicate chunks
        for chunk in chunks_to_delete:
            session.delete(chunk)

        session.commit()

        st.success(f"Cleaned database: Removed {len(chunks_to_delete)} duplicate chunks")


def find_relevant_chunks(
        query: str,
        client: OpenAI,
        k: int = 3
) -> Sequence[Chunk]:
    """Find the k most relevant chunks using semantic similarity via embeddings.

    This function converts the query into an embedding vector and finds the most
    semantically similar chunks in the database, regardless of exact word matches.
    This allows for natural language understanding where different words with
    similar meanings can match, and context is taken into account.
    """
    query_embedding = create_embedding(query, client)

    with open_session() as session:
        # Use pure vector similarity search without any text matching
        chunks = session.exec(
            select(Chunk)
            .where(Chunk.is_embedded == True)  # noqa: E712
            .order_by(Chunk.embedding.l2_distance(query_embedding))
            .limit(k)
        ).all()

        # Debug information to understand what chunks were found
        st.write(f"\nFound {len(chunks)} semantically relevant chunks")

        for i, chunk in enumerate(chunks):
            st.write(f"\nChunk {i + 1}:")
            st.write(f"Title: {chunk.chunk_title}")
            st.write(f"Page: {chunk.page_number + 1}")
            st.write("Content:")
            st.write(chunk.chunk_content)

        return chunks


def generate_answer(
        query: str,
        client: OpenAI,
        k: int = 3
) -> str:
    """Generate an answer for a query using relevant chunks and GPT-4.

    This function takes chunks found through embedding similarity search and uses
    GPT-4 to generate a coherent answer based on their content. It's specifically
    optimized for German technical documentation and ensures proper context
    understanding and source attribution.
    """
    try:
        relevant_chunks = find_relevant_chunks(query, client, k)

        if not relevant_chunks:
            return "Es wurden keine relevanten Informationen in den Dokumenten gefunden."

        # Format context with clear section separation and source attribution
        context = "\n\n".join(
            f"""=== Dokumentenauszug ===
Quelle: {chunk.chunk_title}
Seite: {chunk.page_number + 1}

{chunk.chunk_content}

=================="""
            for chunk in relevant_chunks
        )

        # Create a structured prompt that emphasizes using the provided content
        messages = [
            {
                "role": "system",
                "content": """Sie sind ein technischer Experte, der sich auf Bauvorschriften und technische Richtlinien spezialisiert hat.

Ihre Aufgaben:
- Analysieren Sie die bereitgestellten Dokumentenauszüge gründlich
- Beantworten Sie die Frage basierend auf den tatsächlichen Inhalten der Auszüge
- Zitieren Sie die relevanten Quellen mit Seitenzahlen
- Fassen Sie komplexe technische Informationen klar und verständlich zusammen
- Wenn die Auszüge die relevanten Informationen enthalten, geben Sie diese ausführlich wieder
- Antworten Sie immer auf Deutsch und im Kontext technischer Dokumentation

Wichtig: Basieren Sie Ihre Antwort ausschließlich auf den bereitgestellten Dokumentenauszügen."""
            },
            {
                "role": "user",
                "content": f"""Bitte beantworten Sie folgende Frage basierend auf den Dokumentenauszügen:

Frage: {query}

Verfügbare Dokumentenauszüge:
{context}

Anforderungen an die Antwort:
1. Nutzen Sie die Informationen aus den Dokumentenauszügen
2. Zitieren Sie die relevanten Stellen mit Quellenangabe und Seitenzahl
3. Formulieren Sie die Antwort klar und verständlich
4. Falls die Information tatsächlich nicht in den Auszügen zu finden ist, erklären Sie, welche verwandten Informationen verfügbar sind"""
            }
        ]

        # Get response from GPT-4 with temperature 0 for maximum precision
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0
        )

        return response.choices[0].message.content

    except Exception as e:
        error_msg = f"Fehler bei der Antwortgenerierung: {str(e)}"
        st.error(error_msg)
        return error_msg

def verify_content_exists(search_text: str) -> None:
    """Debug function to verify content exists in database."""
    with open_session() as session:
        # Search for chunks containing the text
        chunks = session.exec(
            select(Chunk)
            .where(Chunk.chunk_content.contains(search_text))
        ).all()

        st.write(f"\nDebug: Found {len(chunks)} chunks containing the search text")
        for chunk in chunks:
            st.write(f"\nFound in: {chunk.chunk_title}, Page {chunk.page_number + 1}")
            st.write("Content preview:")
            st.write(chunk.chunk_content[:200])

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
            with st.spinner(f'Processing {uploaded_file.name}...'):
                documents = load_document(temp_path)
                chunks = create_chunks(documents, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                st.sidebar.write(f'Processed: {uploaded_file.name} ({len(chunks)} chunks)')

        if not all_chunks:
            st.warning("No content was extracted from the documents.")
            return

        # Calculate and display embedding cost
        tokens, cost = calculate_embedding_cost(all_chunks)
        st.sidebar.write(f'Embedding cost: ${cost:.4f} ({tokens:,} tokens)')

        # Create embeddings and store in database
        st.write("Creating embeddings...")
        processed_count = create_embeddings(client, all_chunks)

        if processed_count > 0:
            st.success(f'Successfully created {processed_count} embeddings')
        else:
            st.info('No new embeddings needed to be created')

    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")

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
            client = OpenAI(api_key=api_key)
        else:
            client = None

        uploaded_files = st.file_uploader(
            'Upload documents to analyze:',
            accept_multiple_files=True
        )
        if st.button("Clean Database"):
            clean_database()

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

    # In your main function where you handle questions
    if query:
        if not client:
            st.error('Please provide your OpenAI API key to ask questions.')
            st.stop()

        with st.spinner('Finding answer...'):
            # Add debug information
            st.write("Verifying content in database:")
            verify_content_exists("Zur Anwendung dieses Kommentars")

            # Get the answer
            answer = generate_answer(query, client, k)
            st.text_area('Answer:', value=answer, height=200)

        col1, col2 = st.columns(2)
        with col1:
            # In your main processing logic
            if st.button("Process any pending embeddings"):
                with st.spinner("Processing pending embeddings..."):
                    processed = process_pending_embeddings(client)
                    if processed > 0:
                        st.success(f"Processed {processed} pending embeddings")
                    else:
                        st.info("No pending embeddings to process")
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