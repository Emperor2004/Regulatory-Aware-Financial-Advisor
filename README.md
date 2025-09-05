# RAFAI: Regulatory-Aware Financial Advisor

RAFAI is a chatbot that provides personal finance advice grounded in Indian regulatory evidence from FIU and SEBI. It uses the Gemini API for Retrieval-Augmented Generation (RAG) to deliver cited answers, a Compliance Risk score, and an actionable checklist.

## Core Features
* **Chat Interface:** Ask free-text financial compliance questions.
* **Evidence-Based Answers:** Gemini generates answers based *only* on provided FIU/SEBI text snippets.
* **Compliance Risk Score:** A deterministic rules engine provides an explainable Low/Medium/High risk score.
* **Action Checklist:** Provides clear, downloadable next steps for the user.

## Tech Stack
* **LLM & Embeddings:** Google Gemini API (`gemini-2.5-flash`, `text-embedding-004`)
* **Vector Database:** FAISS (in-memory)
* **UI:** Streamlit
* **Backend:** Python

## How to Run
1.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/Emperor2004/Regulatory-Aware-Financial-Advisor.git
    cd Regulatory-Aware-Financial-Advisor
    pip install -r requirements.txt
    ```
2.  **Set your Gemini API Key:**
    ```bash
    export GEMINI_API_KEY='YOUR_API_KEY'
    ```
3.  **Run the scraper to fetch FIU/SEBI documents:**
    ```bash
    python scraper.py
    ```
3.  **Build the vector index (after scraping):**
    ```bash
    python build_index.py
    ```
4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Demo Script 
1.  Open the Streamlit app.
2.  Use the default question: "I want to transfer 20,00,000 to a crypto exchange in Singapore. Do I need to report?"
3.  Click "Get Advice".
4.  Show the generated answer, the "High" compliance risk, the reason, the downloadable checklist, and the evidence snippets retrieved from the FIU document.