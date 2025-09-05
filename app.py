import os
import re
import faiss
import pickle
import streamlit as st
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAFAI | Regulatory-Aware Financial Advisor",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ENVIRONMENT VARIABLES AND API KEY ---
load_dotenv()
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, AttributeError):
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except (KeyError, AttributeError):
        st.error("🚨 Gemini API Key not found! Please set it in your environment variables or Streamlit secrets.", icon="🔥")
        st.stop()


# --- CORE LOGIC: RAG, GENERATION, AND RISK ANALYSIS ---

@st.cache_resource
def load_faiss_index():
    """Loads the FAISS index and chunk data from disk."""
    try:
        index = faiss.read_index("vector_index.faiss")
        with open("chunks_data.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except FileNotFoundError:
        st.error("🚨 Index files (vector_index.faiss, chunks_data.pkl) not found. Please run the indexing script first.", icon="🔥")
        return None, None

def get_query_embedding(query: str) -> np.ndarray:
    """Generates a vector embedding for the user's query."""
    result = genai.embed_content(model="models/text-embedding-004", content=query)
    return np.array(result['embedding']).reshape(1, -1)

def retrieve_top_passages(query_embedding: np.ndarray, index: faiss.Index, chunks: list, k: int = 3) -> list:
    """Retrieves the top-k most relevant passages from the FAISS index."""
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_response(query: str, passages: list) -> str:
    """Generates a response using the Gemini model based on the retrieved passages."""
    evidence_text = "\n\n---\n\n".join(
        [f"Evidence from document '{p.get('doc_id', 'Unknown')}':\n{p['text']}" for p in passages]
    )

    prompt = f"""
    You are RAFAI, a helpful and friendly regulatory-aware financial assistant for India. Your audience is the general public, so explain concepts clearly and simply.

    **Primary Instructions:**
    1.  **Analyze the User's Question:** Carefully understand the user's situation based on their question.
    2.  **Use the Evidence:** Base your answer *strictly* on the provided evidence text below. Do not use outside knowledge.
    3.  **Produce Four Sections in Your Output:**
        * **Short Answer:** A brief, 2-4 sentence summary answering the user's question directly.
        * **ComplianceRisk:** Assess the risk as 'Low', 'Medium', or 'High' based *only* on the evidence. Provide a short reason.
        * **ActionChecklist:** Create a list of 2-3 simple, actionable steps the user should take.
        * **Evidence Snippets:** Cite 1-2 direct quotes from the evidence that support your answer, along with their source document ID.
    
    **Formatting Rule:** Do NOT use markdown formatting like asterisks for bolding. Output in plain text.

    ---
    **Evidence:**
    {evidence_text}
    ---
    **User's Question:** {query}
    ---
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        # Clean up any residual markdown
        return response.text.replace('**', '')
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Sorry, I was unable to generate a response. Please try again."

def apply_deterministic_rules(query: str) -> (str, str):
    """
    Applies robust, explainable rules to identify clear high-risk scenarios
    based on the latest Indian financial regulations.
    """
    risk = "Low"
    reasons = []
    query_lower = query.lower()
    
    money_pattern = r'₹?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore)?'
    matches = re.findall(money_pattern, query_lower)
    
    # Extract the highest numerical value mentioned in the query
    max_amount = 0
    for amount_str, unit in matches:
        amount = float(amount_str.replace(',', ''))
        if unit.lower() in ['lakh', 'lac']:
            amount *= 100000
        elif unit.lower() == 'crore':
            amount *= 10000000
        if amount > max_amount:
            max_amount = amount

    # Rule 1: High-value cash/equivalent transactions (Threshold: ₹10,00,000)
    if max_amount >= 1000000:
        risk = "High"
        reasons.append(f"Transaction amount (₹{max_amount:,.0f}) exceeds the ₹10 lakh reporting threshold for single transactions.")

    # Rule 2: Cross-border wire transfers (Threshold: ₹5,00,000)
    cross_border_keywords = ["international", "abroad", "overseas", "cross-border", "wire transfer", "remittance", "singapore", "usa", "uk"]
    if any(keyword in query_lower for keyword in cross_border_keywords) and max_amount > 500000:
        risk = "High"
        reasons.append(f"Cross-border wire transfer (₹{max_amount:,.0f}) exceeds the ₹5 lakh reporting threshold.")

    # Rule 3: Crypto transfers - High risk regardless of amount
    crypto_keywords = ["crypto", "virtual asset", "bitcoin", "ethereum", "vda", "nft"]
    if any(keyword in query_lower for keyword in crypto_keywords):
        risk = "High"
        reasons.append("Involves crypto/virtual asset transfer, which has specific reporting requirements under PMLA.")

    # Remove duplicate reasons if any
    unique_reasons = list(dict.fromkeys(reasons))
    return risk, " ".join(unique_reasons) if unique_reasons else "No high-risk keywords or thresholds detected."


def parse_llm_output(output_text: str) -> dict:
    """
    Parses the structured output from the LLM into a dictionary.
    Uses regex for more robust parsing.
    """
    parsed_data = {}
    
    patterns = {
        'answer': r"Short Answer:(.*?)(?=ComplianceRisk:|ActionChecklist:|Evidence Snippets:|$)",
        'risk': r"ComplianceRisk:(.*?)(?=ActionChecklist:|Evidence Snippets:|$)",
        'checklist': r"ActionChecklist:(.*?)(?=Evidence Snippets:|$)",
        'snippets': r"Evidence Snippets:(.*)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
        # Clean up the text by removing asterisks and stripping whitespace
        parsed_data[key] = match.group(1).replace('**', '').strip() if match else "Not found."

    risk_text = parsed_data.get('risk', '')
    if 'high' in risk_text.lower():
        parsed_data['risk_level'] = 'High'
    elif 'medium' in risk_text.lower():
        parsed_data['risk_level'] = 'Medium'
    else:
        parsed_data['risk_level'] = 'Low'
        
    return parsed_data

# --- STREAMLIT UI ---

st.title("RAFAI: Regulatory-Aware Financial Advisor 🇮🇳")
st.caption("Your AI assistant for navigating Indian financial regulations. Powered by Google Gemini.")

index, chunks = load_faiss_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("Ask a Compliance Question")
    default_question = "I want to transfer 20,00,000 to a crypto exchange in Singapore. Do I need to report this?"
    user_query = st.text_area("Your Question:", default_question, height=120, key="user_query_input")
    
    spinner_placeholder = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Get Advice", type="primary", use_container_width=True):
            if not user_query:
                st.warning("Please enter a question.")
            elif index is None:
                st.error("Cannot proceed, index is not loaded.")
            else:
                with spinner_placeholder, st.spinner("Analyzing regulations..."):
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    query_embedding = get_query_embedding(user_query)
                    top_passages = retrieve_top_passages(query_embedding, index, chunks)
                    llm_response_text = generate_response(user_query, top_passages)
                    parsed_llm_response = parse_llm_output(llm_response_text)
                    rule_risk, rule_reason = apply_deterministic_rules(user_query)
                    llm_risk_level = parsed_llm_response.get('risk_level', 'Low')
                    final_risk = rule_risk if rule_risk == "High" else llm_risk_level
                    
                    final_checklist = parsed_llm_response.get('checklist', 'No checklist provided.')
                    if final_risk == "High":
                        high_risk_actions = [
                            "Critical: Submit a mandatory report (e.g., CTR, STR) to the FIU-IND.",
                            "Critical: Immediately notify your bank’s designated compliance officer of this transaction."
                        ]
                        # Prepend high-risk actions to the checklist
                        final_checklist = "\n".join(high_risk_actions) + "\n" + final_checklist
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": parsed_llm_response.get('answer'),
                        "final_risk": final_risk,
                        "llm_risk": llm_risk_level,
                        "llm_risk_reason": parsed_llm_response.get('risk'),
                        "rule_risk": rule_risk,
                        "rule_reason": rule_reason,
                        "checklist": final_checklist,
                        "snippets": parsed_llm_response.get('snippets'),
                        "evidence": top_passages,
                    }
                    st.session_state.messages.append(assistant_message)
    
    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Disclaimer: This is an informational demo and not legal or financial advice. Always consult a qualified professional.")


# --- Main Dashboard Interface ---
if not st.session_state.messages:
    st.info("Ask a question in the sidebar to get started and see the analysis here!")
else:
    # Get the last user message and the last assistant message
    last_user_msg = next((msg for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
    last_asst_msg = next((msg for msg in reversed(st.session_state.messages) if msg["role"] == "assistant"), None)

    if not last_user_msg or not last_asst_msg:
        st.warning("Could not find the last interaction. Please ask a new question.")
        st.stop()


    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("Your Question")
        st.info(last_user_msg['content'])
        
        st.subheader("RAFAI's Advice")
        with st.container(border=True):
             st.markdown(last_asst_msg['content'])
             st.divider()
             st.markdown(last_asst_msg.get("snippets", "No snippets available."))

    with col2:
        st.subheader("Compliance Analysis")
        final_risk = last_asst_msg.get("final_risk", "N/A")
        if final_risk == "High":
            st.error(f"**Risk Level: {final_risk}**", icon="🚨")
        elif final_risk == "Medium":
            st.warning(f"**Risk Level: {final_risk}**", icon="⚠️")
        else:
            st.success(f"**Risk Level: {final_risk}**", icon="✅")
        
        primary_reason = last_asst_msg.get('rule_reason') if last_asst_msg.get('rule_risk') == 'High' else last_asst_msg.get('llm_risk_reason')
        st.caption(f"**Reason:** {primary_reason}")
        st.divider()

        st.markdown("**Action Checklist**")
        checklist_text = last_asst_msg.get("checklist", "No checklist available.")
        st.markdown(checklist_text)
        st.download_button(
            label="⬇️ Download Checklist",
            data=checklist_text.encode('utf-8'),
            file_name="ActionChecklist.txt",
            mime="text/plain",
        )
        st.divider()

        with st.expander("Show Compliance Breakdown"):
            st.markdown(f"**1. Deterministic Rule Check:** `{last_asst_msg.get('rule_risk')}`")
            st.caption(f"Details: {last_asst_msg.get('rule_reason')}")
            st.markdown(f"**2. AI Model Assessment:** `{last_asst_msg.get('llm_risk')}`")
            st.caption(f"Details: {last_asst_msg.get('llm_risk_reason')}")

        with st.expander("Show Retrieved Evidence & Provenance"):
            for p in last_asst_msg.get("evidence", []):
                st.markdown(f"**Source:** `{p.get('doc_id', 'Unknown')}`")
                st.markdown(f"> {p.get('text', 'Content not available.')}")
                source_url = p.get('source_url')
                if source_url:
                    st.markdown(f"**Original URL:** [{source_url}]({source_url})")
                st.markdown("---")

