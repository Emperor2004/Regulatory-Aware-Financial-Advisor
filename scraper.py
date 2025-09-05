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
        # Get the PDF content
        response = requests.get(pdf_url, timeout=20)
        response.raise_for_status() # Raise an exception for bad status codes

        # Read the PDF from the in-memory content
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                # Add a space to handle text that spans pages
                full_text += page.extract_text() + " "
            return full_text
    except requests.exceptions.RequestException as e:
        print(f"  !! Error fetching PDF {pdf_url}: {e}")
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
        response = requests.get(urljoin(BASE_URL, 'index.html'), timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the 'What's New' section and then the list items within it
        # Note: Website structure can change. This selector may need updating.
        whats_new_section = soup.find('div', id='whatsnew-list')
        if not whats_new_section:
            print("Could not find the 'What's New' section. The website structure may have changed.")
            return []

        links = whats_new_section.find_all('a', href=True)
        print(f"Found {len(links)} links in 'What's New'. Processing the first {limit}...")

        for link in links[:limit]:
            doc_title = link.get_text(strip=True)
            doc_url = urljoin(BASE_URL, link['href'])

            # We are primarily interested in PDFs which contain detailed notices.
            if doc_url.lower().endswith('.pdf'):
                content = extract_text_from_pdf(doc_url)
                if content:
                    documents.append({
                        "doc_id": doc_title,
                        "content": content,
                        "source_url": doc_url
                    })
            else:
                print(f"  -> Skipping non-PDF link: {doc_url}")

    except requests.exceptions.RequestException as e:
        print(f"!! Critical error fetching the main FIU page: {e}")
    
    return documents

if __name__ == "__main__":
    # Scrape the data
    scraped_docs = scrape_fiu_whats_new(limit=3) # Limit to 3 documents to be fast
    
    # Save the scraped data to a JSON file
    if scraped_docs:
        with open("scraped_data.json", "w", encoding='utf-8') as f:
            json.dump(scraped_docs, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Successfully scraped and saved {len(scraped_docs)} documents to scraped_data.json")
    else:
        print("\n❌ No documents were scraped. Please check the scraper logic and website structure.")