import requests
from bs4 import BeautifulSoup
import pdfplumber
import io
import json
from urllib.parse import urljoin

# Base URL for the FIU India website
BASE_URL = "https://fiuindia.gov.in/"

def extract_text_from_pdf(pdf_url):
    """Downloads a PDF from a URL and extracts its text content."""
    print(f"  -> Processing PDF: {pdf_url}")
    try:
        response = requests.get(pdf_url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + " " if page.extract_text() else ""
            return full_text
    except Exception as e:
        print(f"  !! Error processing PDF {pdf_url}: {e}")
    return ""

def scrape_fiu_whats_new(limit=3):
    """
    Scrapes the 'What's New' section of the FIU India website for recent documents.
    """
    print("🚀 Starting scraper for fiuindia.gov.in...")
    documents = []
    
    try:
        response = requests.get(urljoin(BASE_URL, 'index.html'), timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        print(f"Successfully fetched page with title: '{soup.title.string}'")

        # --- KEY CHANGE IS HERE ---
        # The site uses a class named 'whatsnew' for the containing div.
        # We use 'class_' because 'class' is a reserved keyword in Python.
        whats_new_section = soup.find('div', class_='whatsnew')
        
        if not whats_new_section:
            print("❌ Could not find the 'What's New' section with class 'whatsnew'. The website structure may have changed.")
            return []

        links = whats_new_section.find_all('a', href=True)
        print(f"Found {len(links)} links in 'What's New'. Processing the first {limit}...")

        for link in links[:limit]:
            doc_title = link.get_text(strip=True)
            doc_url = urljoin(BASE_URL, link['href'])

            if doc_url.lower().endswith('.pdf'):
                content = extract_text_from_pdf(doc_url)
                if content:
                    documents.append({
                        "doc_id": doc_title if doc_title else f"Document from {doc_url.split('/')[-1]}",
                        "content": content,
                        "source_url": doc_url
                    })
            else:
                print(f"  -> Skipping non-PDF link: {doc_url}")

    except requests.exceptions.RequestException as e:
        print(f"!! Critical error fetching the main FIU page: {e}")
    
    return documents

if __name__ == "__main__":
    scraped_docs = scrape_fiu_whats_new(limit=3)
    
    if scraped_docs:
        with open("scraped_data.json", "w", encoding='utf-8') as f:
            json.dump(scraped_docs, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Successfully scraped and saved {len(scraped_docs)} documents to scraped_data.json")
    else:
        print("\n❌ No documents were scraped. Please check the scraper logic and website structure.")