import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import os
import re

# Base URL for PubMed
base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "/?term=(p42es017198[Grant+Number])+OR+(p42+es017198[Grant+Number])&sort=date"
current_url = base_url + search_url

# Directory to save the tokenized text files
output_dir = r"C:\Users\musak\OneDrive\Datascraping\Datascraping\tokenized_text"
os.makedirs(output_dir, exist_ok=True)

def scrape_page(url):
    """Scrape a page for articles and follow links to their full texts."""
    print("Scraping URL: " + url)
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        paper_links = get_paper_links(soup)
        for paper_link in paper_links:
            extract_and_tokenize(paper_link)
        next_page_url = get_next_page_url(soup, url)
        if next_page_url:
            scrape_page(next_page_url)
        else:
            print("No more pages to scrape.")
    except requests.RequestException as e:
        print(f"Failed to scrape the page: {e}")

def get_paper_links(soup):
    """Extracts paper links from a PubMed search result page."""
    try:
        links = soup.find_all('a', class_='docsum-title')
        paper_links = [urljoin(base_url, link['href']) for link in links]
        return paper_links
    except Exception as e:
        print(f"Failed to get paper links: {e}")
        return []

def get_next_page_url(soup, current_url):
    """Finds the URL for the next page of search results."""
    try:
        next_page_url = None
        next_page_button = soup.find('button', class_='next-page-btn')
        if next_page_button and 'disabled' not in next_page_button.attrs:
            parsed_url = urlparse(current_url)
            query_params = parse_qs(parsed_url.query)
            current_page = int(query_params.get('page', [1])[0])
            query_params['page'] = [str(current_page + 1)]
            next_page_url = urlunparse(parsed_url._replace(query=urlencode(query_params, doseq=True)))
        return next_page_url
    except Exception as e:
        print(f"Failed to get next page URL: {e}")
        return None

def extract_and_tokenize(paper_url):
    """Extracts sections from a paper, tokenizes them, and writes tokens into text files."""
    try:
        response = requests.get(paper_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', class_='heading-title')
        title = title_tag.text.strip() if title_tag else 'N/A'
        sanitized_title = sanitize_title(title)
        print(f"Article Title: {title}")
        print(f"Article URL: {paper_url}")
        
        full_text_url_tag = soup.find('a', class_='link-item pmc')
        if full_text_url_tag:
            full_text_url = full_text_url_tag['href']
            full_text_url = urljoin(base_url, full_text_url)
            print(f"  Full Text URL: {full_text_url}")
            tokenize_and_save(full_text_url, sanitized_title)
        else:
            print("  Full text not available")
    except requests.RequestException as e:
        print(f"Failed to retrieve article details: {e}")
    except Exception as e:
        print(f"An error occurred while extracting details: {e}")

def tokenize_and_save(full_text_url, sanitized_title):
    """Tokenizes the full text and saves meaningful English words into a single text file."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(full_text_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text_content = soup.get_text()
        tokens = tokenize_text(text_content)
        
        # Write tokens to a single text file
        file_name = f"{sanitized_title}_tokens.txt"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for section, section_tokens in tokens.items():
                f.write(f"=== {section.capitalize()} ===\n")
                f.write('\n'.join(section_tokens))
                f.write('\n\n')
                
        print(f"Tokens saved to '{file_path}'")
    except requests.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
    except Exception as e:
        print(f"An error occurred while parsing the full text: {e}")

def tokenize_text(text):
    """Tokenizes text into meaningful English words."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Filter out non-meaningful tokens
    meaningful_tokens = [token for token in tokens if len(token) > 2 and token.isalpha()]
    
    # Group tokens by sections (you can customize these section definitions)
    section_tokens = {
        'introduction': [],
        'methods': [],
        'results': [],
        'discussion': [],
        'conclusion': [],
        'abstract': []
    }
    
    # Define regex patterns for section identification
    section_patterns = {
        'introduction': re.compile(r'\bintroduction\b', re.IGNORECASE),
        'methods': re.compile(r'\bmethods?\b', re.IGNORECASE),
        'results': re.compile(r'\bresults?\b', re.IGNORECASE),
        'discussion': re.compile(r'\bdiscussions?\b', re.IGNORECASE),
        'conclusion': re.compile(r'\bconclusions?\b', re.IGNORECASE),
        'abstract': re.compile(r'\babstract\b', re.IGNORECASE),
    }
    
    # Iterate over tokens and assign them to respective sections
    current_section = None
    for token in meaningful_tokens:
        for section, pattern in section_patterns.items():
            if pattern.search(token):
                current_section = section
                break
        if current_section:
            section_tokens[current_section].append(token)
    
    return section_tokens

def sanitize_title(title):
    """Sanitizes the title to make it suitable for file names."""
    invalid_chars = r'<>:"/\\|?*'
    for char in invalid_chars:
        title = title.replace(char, "")
    title = re.sub(r'\s+', '_', title.strip())
    return title

if __name__ == "__main__":
    try:
        scrape_page(current_url)
    except Exception as e:
        print(f"An error occurred during execution: {e}")

