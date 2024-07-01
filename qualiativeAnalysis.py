import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import re
import os

# Base URL for PubMed
base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "/?term=(p42es017198[Grant+Number])+OR+(p42+es017198[Grant+Number])&sort=date"
current_url = base_url + search_url

# Function to scrape each page
def scrape_page(url):
    print("Scraping URL: " + url)
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        paper_links = get_paper_links(soup)
        for paper_link in paper_links:
            extract_and_check_sections(paper_link)
        next_page_url = get_next_page_url(soup, url)
        if next_page_url:
            scrape_page(next_page_url)
        else:
            print("No more pages to scrape.")
    except requests.RequestException as e:
        print(f"Failed to scrape the page: {e}")

# Function to extract paper links from the current page
def get_paper_links(soup):
    try:
        links = soup.find_all('a', class_='docsum-title')
        paper_links = [urljoin(base_url, link['href']) for link in links]
        return paper_links
    except Exception as e:
        print(f"Failed to get paper links: {e}")
        return []

# Function to get URL of the next page
def get_next_page_url(soup, current_url):
    try:
        next_page_url = None
        next_page_button = soup.find('button', class_='next-page-btn')
        if next_page_button and 'disabled' not in next_page_button.attrs:
            # Construct the URL for the next page
            parsed_url = urlparse(current_url)
            query_params = parse_qs(parsed_url.query)
            current_page = int(query_params.get('page', [1])[0])
            query_params['page'] = [str(current_page + 1)]
            next_page_url = urlunparse(parsed_url._replace(query=urlencode(query_params, doseq=True)))
        return next_page_url
    except Exception as e:
        print(f"Failed to get next page URL: {e}")
        return None

# Function to extract and check sections from a paper
def extract_and_check_sections(paper_url):
    try:
        response = requests.get(paper_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', class_='heading-title')
        title = title_tag.text.strip() if title_tag else 'N/A'
        print(f"Article Title: {title}")
        print(f"Article URL: {paper_url}")
        
        # Check if full text link is available
        full_text_url = soup.find('a', class_='link-item pmc')
        if full_text_url:
            full_text_url = full_text_url['href']
            full_text_url = urljoin(base_url, full_text_url)
            print(f"  Full Text URL: {full_text_url}")
            check_full_text_sections(full_text_url, title)
        else:
            print("  Full text not available")
            # If full text link is not available, report all sections as Not Available
            sections_presence = {section: False for section in [
                "Abstract", "Introduction", "Materials", "Methods",
                "Methodology", "Results", "Discussion", "Conclusion"
            ]}
            prompt_user_with_sections(title, sections_presence, None)
        
        print("---")
    except requests.RequestException as e:
        print(f"Failed to retrieve article details: {e}")
    except Exception as e:
        print(f"An error occurred while extracting details: {e}")

# Function to check sections presence in the full text of the article
def check_full_text_sections(full_text_url, paper_title):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        if not full_text_url:
            print("Full text link not found. Skipping...")
            sections_presence = {section: False for section in [
                "Abstract", "Introduction", "Materials", "Methods",
                "Methodology", "Results", "Discussion", "Conclusion"
            ]}
        else:
            response = requests.get(full_text_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for the presence of sections
            sections_presence = check_sections_from_html(soup)
            
            # Tokenize and filter text from each section
            for section, is_present in sections_presence.items():
                if is_present:
                    section_text = extract_section_text(soup, section)
                    if section_text:
                        print(f"Tokens for '{section}':")
                        tokens = tokenize_and_filter_text(section_text)
                        print(tokens[:50])  # Print first 50 tokens (for demonstration)
                        
                        # Write tokens to file
                        write_tokens_to_file(tokens, paper_title, section)
        
        # Prompt user with sections and full text URL
        prompt_user_with_sections(paper_title, sections_presence, full_text_url)
        
    except requests.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
    except Exception as e:
        print(f"An error occurred while parsing the full text: {e}")

# Function to check presence of sections in the HTML of the article
def check_sections_from_html(soup):
    sections_presence = {
        "Abstract": False,
        "Introduction": False,
        "Materials": False,
        "Methods": False,
        "Methodology": False,
        "Results": False,
        "Discussion": False,
        "Conclusion": False
    }

    section_patterns = {
        "Abstract": re.compile(r'\babstract\b', re.IGNORECASE),
        "Introduction": re.compile(r'\bintroduction\b', re.IGNORECASE),
        "Materials": re.compile(r'\bmaterials\b', re.IGNORECASE),
        "Methods": re.compile(r'\bmethods\b', re.IGNORECASE),
        "Methodology": re.compile(r'\bmethodology\b', re.IGNORECASE),
        "Results": re.compile(r'\bresults\b', re.IGNORECASE),
        "Discussion": re.compile(r'\bdiscussions?\b', re.IGNORECASE),
        "Conclusion": re.compile(r'\bconclusions?\b', re.IGNORECASE),
    }

    headers = soup.find_all(re.compile('^h[1-6]$'))
    for header in headers:
        header_text = header.get_text().strip()
        for section, pattern in section_patterns.items():
            if pattern.search(header_text):
                sections_presence[section] = True

    return sections_presence

# Function to extract text content of a section from the HTML
def extract_section_text(soup, section_name):
    try:
        section_tag = soup.find(text=re.compile(r'\b{}\b'.format(section_name), re.IGNORECASE))
        if section_tag:
            section_text = section_tag.find_parent().find_next_sibling()
            if section_text:
                return section_text.get_text(separator=' ', strip=True)
        return None
    except Exception as e:
        print(f"Error extracting {section_name} section: {e}")
        return None

# Function to tokenize and filter text
def tokenize_and_filter_text(text):
    # Replace non-alphanumeric characters with spaces and split into tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Example filtering: Keep tokens longer than 2 characters
    tokens = [token for token in tokens if len(token) > 2]
    return tokens

# Function to write tokens to file
def write_tokens_to_file(tokens, paper_title, section_name):
    try:
        folder_name = "tokenized_text"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        file_name = f"{folder_name}/{paper_title}_{section_name}_tokens.txt"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tokens))
        print(f"Tokens written to: {file_name}")
    except Exception as e:
        print(f"Error writing tokens to file: {e}")

# Function to prompt user with sections found
def prompt_user_with_sections(paper_title, sections_presence, full_text_url):
    print(f"Sections found in '{paper_title}':")
    for section, is_present in sections_presence.items():
        status = "Available" if is_present else "Not Available"
        print(f"  {section}: {status}")
    if full_text_url:
        print(f"Full Text URL: {full_text_url}\n")
    else:
        print("Full Text URL: Not available\n")

if __name__ == "__main__":
    scrape_page(current_url)

