import os
import faiss
import pickle
import streamlit as st
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()
# Configure the Gemini API client
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# --- RAG AND GENERATION LOGIC ---

def get_query_embedding(query):
    """Generates an embedding for the user's query."""
    result = genai.embed_content(model="models/text-embedding-004", content=query)
    return np.array(result['embedding']).reshape(1, -1) # Reshape for FAISS search

def retrieve_top_passages(query_embedding, index, chunks, k=3):
    """Retrieves the top-k most relevant passages from the FAISS index."""
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_response(query, passages):
    """Generates a response using Gemini based on the retrieved passages."""
    
    evidence_text = "\n\n".join(
        [f"Evidence from {p['doc_id']}:\n{p['text']}" for p in passages]
    )

    # The prompt template has been updated to fix the SyntaxError
    prompt = f"""
    You are RAFAI, a regulatory-aware financial assistant. Use ONLY the evidence passages below to answer the user.
    Follow these 4 tasks precisely:
    1) Provide a short answer (2-4 sentences).
    2.1) Cite up to 2 relevant evidence snippets from the passages.
    
    # --- THIS IS THE CORRECTED LINE ---
    2.2) For each snippet, include the document ID as a citation like ``. 
    
    3) Output a 'ComplianceRisk' score (Low, Medium, or High) with a one-line reason.
    4) Output an 'ActionChecklist' with up to 3 short, actionable tasks (who, what, when).

    ---
    Evidence Passages:
    {evidence_text}
    ---
    User Question: {query}
    ---
    """
    
    # Call Gemini API
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- DETERMINISTIC RISK RULES ---
def apply_deterministic_rules(query, llm_output):
    """Applies simple, explainable rules to assign a final risk score."""
    risk = "Low" # Default risk
    reasons = []

    # Rule 1: Transaction Amount [cite: 84]
    if "20 lakh" in query or "20,00,000" in query:
        risk = "High"
        reasons.append("Transaction amount (20 Lakh) exceeds the 10 Lakh INR reporting threshold.")
    
    # Rule 2: Crypto Keyword [cite: 85]
    if "crypto" in query.lower():
        risk = "High"
        reasons.append("Involves cross-border crypto transfer, which requires mandatory reporting.")
    
    return risk, " ".join(reasons)


# --- STREAMLIT UI SETUP --- [cite: 88, 89]

st.set_page_config(page_title="RAFAI Demo", layout="wide")
st.title("RAFAI: Regulatory-Aware Financial Advisor 🇮🇳")

# Load the FAISS index and chunk data once
@st.cache_resource
def load_index():
    import subprocess

    # If index/chunks not found, auto-run scraper + build_index
    if not (os.path.exists("vector_index.faiss") and os.path.exists("chunks_data.pkl")):
        st.warning("Index files not found. Running scraper and build_index...")

        try:
            # Run scraper.py
            subprocess.run(["python", "scraper.py"], check=True)

            # Run build_index.py
            subprocess.run(["python", "build_index.py"], check=True)
        except Exception as e:
            st.error(f"Failed to build index automatically: {e}")
            return None, None

    # After ensuring files exist, load them
    try:
        index = faiss.read_index("vector_index.faiss")
        with open("chunks_data.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None, None

index, chunks = load_index()

if index is None:
    st.error("Index files not found. Please run `build_index.py` first.")
else:
    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main chat interface
    st.sidebar.header("Ask a Compliance Question")
    
    # Use the example question from the document as a placeholder [cite: 106]
    default_question = "I want to transfer 20,00,000 to a crypto exchange in Singapore. Do I need to report?"
    user_query = st.sidebar.text_area("Your Question:", default_question, height=100)
    
    if st.sidebar.button("Get Advice"):
        if user_query:
            with st.spinner("Analyzing regulations..."):
                # 1. Embed the user query
                query_embedding = get_query_embedding(user_query)
                
                # 2. Retrieve relevant passages
                top_passages = retrieve_top_passages(query_embedding, index, chunks)
                
                # 3. Generate a response from Gemini
                llm_response_text = generate_response(user_query, top_passages)

                # 4. Apply deterministic rules for final risk score [cite: 87]
                final_risk, risk_reason = apply_deterministic_rules(user_query, llm_response_text)

                # Store and display conversation
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": llm_response_text,
                    "risk": final_risk,
                    "reason": risk_reason,
                    "evidence": top_passages
                })
        else:
            st.sidebar.warning("Please enter a question.")

    # Display the conversation
    if st.session_state.messages:
        # Get the last interaction
        last_user_msg = st.session_state.messages[-2]
        last_asst_msg = st.session_state.messages[-1]

        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.info(f"**Your Question:**\n\n{last_user_msg['content']}")
            st.success(f"**RAFAI's Advice:**\n\n{last_asst_msg['content']}")

        with col2:
            st.subheader("Compliance Analysis")
            
            # Compliance Risk Gauge [cite: 89]
            if last_asst_msg['risk'] == "High":
                st.error(f"**Risk Level: {last_asst_msg['risk']}**")
            elif last_asst_msg['risk'] == "Medium":
                st.warning(f"**Risk Level: {last_asst_msg['risk']}**")
            else:
                st.success(f"**Risk Level: {last_asst_msg['risk']}**")
            
            st.caption(f"**Reason:** {last_asst_msg['reason']}")
            
            # Action Checklist Download [cite: 89]
            checklist_text = last_asst_msg['content'].split("ActionChecklist:")[-1].strip()
            st.download_button(
                label="⬇️ Download Action Checklist",
                data=checklist_text,
                file_name="ActionChecklist.txt",
                mime="text/plain"
            )

            # Evidence Panel
            with st.expander("Show Retrieved Evidence & Provenance", expanded=True):
                for passage in last_asst_msg['evidence']:
                    st.markdown(f"> **Source Document:** `{passage['doc_id']}`")
                    st.markdown(f"> **Original URL:** [Link]({passage['source_url']})") # <-- ADD THIS LINE
                    st.markdown(f"**Retrieved Content:**\n\n ...{passage['text']}...")
                    st.divider()

st.sidebar.markdown("---")
st.sidebar.caption("Disclaimer: This is an informational demo and not legal/financial advice.")