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
    You are RAFAI, a regulatory-aware financial assistant.
    Important rule: DO NOT alter numbers or amounts from the user’s question. Repeat them exactly as given.

    Tasks:
    1) Provide a short answer (2-4 sentences).
    2) Cite up to 2 relevant evidence snippets (with doc IDs).
    3) Output a 'ComplianceRisk' (Low, Medium, High) with reason.
    4) Output an 'ActionChecklist' with up to 3 short tasks.

    ---
    Evidence:
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
def apply_deterministic_rules(query):
    """Applies explainable rules to assign a risk score."""
    risk = "Low"
    reasons = []

    # Rule 1: High-value transactions
    if "20 lakh" in query or "20,00,000" in query or "10,00,000" in query:
        risk = "High"
        reasons.append("Transaction exceeds the ₹10 lakh reporting threshold.")

    # Rule 2: Crypto transfers
    if "crypto" in query.lower():
        risk = "High"
        reasons.append("Involves crypto transfer, which requires mandatory reporting.")

    return risk, " ".join(reasons)

# --- LLM RISK ---
def extract_llm_risk(output_text):
    """Parse the Gemini response to extract ComplianceRisk."""
    for line in output_text.splitlines():
        if "ComplianceRisk" in line:
            if "High" in line:
                return "High"
            elif "Medium" in line:
                return "Medium"
            elif "Low" in line:
                return "Low"
    return "Unknown"

# --- REDINE CHECKLIST ---
def refine_checklist(llm_output, final_risk):
    checklist_text = llm_output.split("ActionChecklist:")[-1].strip()
    tasks = checklist_text.split("\n")

    if final_risk == "High":
        tasks.insert(0, "Submit CTR to FIU-IND within 7 days.")
        tasks.insert(1, "Notify your bank’s compliance officer immediately.")
    return "\n".join(tasks)


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="RAFAI Demo", layout="wide")
st.title("RAFAI: Regulatory-Aware Financial Advisor 🇮🇳")

# Load the FAISS index and chunk data once
@st.cache_resource
def load_index():
    try:
        index = faiss.read_index("vector_index.faiss")
        with open("chunks_data.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except FileNotFoundError:
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
    
    # Use the example question from the document as a placeholder
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

                # 4. Extract LLM risk
                llm_risk = extract_llm_risk(llm_response_text)

                # 5. Apply deterministic rules
                rule_risk, risk_reason = apply_deterministic_rules(user_query)

                # 6. Final risk = deterministic takes precedence
                final_risk = rule_risk if rule_risk != "Low" else llm_risk

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

            st.markdown(f"**LLM Assessment:** {llm_risk}")
            st.markdown(f"**Deterministic Rule Risk:** {rule_risk}")
            
            if final_risk == "High":
                st.error(f"**Final Risk Level: {final_risk}**")
            elif final_risk == "Medium":
                st.warning(f"**Final Risk Level: {final_risk}**")
            else:
                st.success(f"**Final Risk Level: {final_risk}**")

            st.caption(f"**Reason:** {risk_reason if risk_reason else 'Based on evidence interpretation.'}")

            # Action Checklist Download
            checklist_text = refine_checklist(llm_response_text, final_risk)
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