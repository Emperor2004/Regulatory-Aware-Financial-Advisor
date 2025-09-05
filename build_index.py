import faiss
import pickle
import json
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np
import streamlit as st

# Configure the Gemini API client
load_dotenv()
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def get_text_chunks_from_json(filepath="scraped_data.json"):
    """Loads scraped data and splits it into chunks."""
    chunks = []
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run scraper.py first.")
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    for doc in documents:
        for chunk in splitter.split_text(doc["content"]):
            chunks.append({
                "text": chunk,
                "doc_id": doc["doc_id"],
                "source_url": doc["source_url"]
            })
    return chunks

def create_embeddings(chunks):
    """Creates Gemini embeddings for a list of text chunks."""
    texts = [chunk["text"] for chunk in chunks]
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=texts)
        return result['embedding']
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return None

def build_and_save_index(embeddings, chunks):
    """Builds a FAISS index and saves it along with the chunk data."""
    if embeddings is None or not embeddings:
        print("No embeddings to process.")
        return

    # Convert the list of embeddings to a NumPy array
    embeddings_np = np.array(embeddings).astype('float32') # <-- 2. CONVERT TO NUMPY ARRAY

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np) # Use the NumPy array here

    faiss.write_index(index, "vector_index.faiss")
    with open("chunks_data.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Index and data saved successfully!")

if __name__ == "__main__":
    print("Starting the indexing process...")
    text_chunks = get_text_chunks_from_json()
    if text_chunks:
        print(f"Created {len(text_chunks)} text chunks from scraped data.")
        gemini_embeddings = create_embeddings(text_chunks)
        if gemini_embeddings:
            build_and_save_index(gemini_embeddings, text_chunks)