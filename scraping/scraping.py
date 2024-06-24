import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import pdfplumber
import re
import os

base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "/?term=(p42es017198[Grant+Number])+OR+(p42+es017198[Grant+Number])&sort=date"
current_url = base_url + search_url

def scrape_page(url):
    print("Scraping URL: " + url)
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        paper_links = get_paper_links(soup, url)
        for paper_link in paper_links:
            extract_and_print_details(paper_link)
        next_page_url = get_next_page_url(soup, url)
        if next_page_url:
            scrape_page(next_page_url)
    except requests.RequestException as e:
        print(f"Failed to scrape the page: {e}")

def get_paper_links(soup, search_url):
    links = soup.find_all('a', class_='docsum-title')
    paper_links = [urljoin(base_url, link['href']) for link in links]
    return paper_links

def get_next_page_url(soup, current_url):
    next_page_url = None
    next_page_button = soup.find('button', class_='next-page-btn')
    if next_page_button:
        parsed_url = urlparse(current_url)
        query_params = parse_qs(parsed_url.query)
        if 'page' in query_params:
            current_page = int(query_params['page'][0])
            next_page = current_page + 1
            query_params['page'] = [str(next_page)]
        else:
            query_params['page'] = ['2']
        next_page_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, urlencode(query_params, doseq=True), parsed_url.fragment))
    return next_page_url

def extract_and_print_details(paper_url):
    try:
        response = requests.get(paper_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', class_='heading-title')
        title = title_tag.text.strip() if title_tag else 'N/A'
        print(f"Article Title: {title}")
        print(f"Article URL: {paper_url}")
        author_tags = soup.find_all('a', class_='full-name')
        authors = ", ".join([author.text for author in author_tags])
        date_tag = soup.find('span', class_='cit')
        publication_date = date_tag.text.strip() if date_tag else 'N/A'
        print(f"  Authors: {authors}")
        print(f"  Publication Date: {publication_date}")
        
     
        full_text_url = soup.find('a', class_='link-item pmc')
        if full_text_url:
            full_text_url = full_text_url['href']
            full_text_url = urljoin(base_url, full_text_url)
            print(f"  Full Text URL: {full_text_url}")
            download_paper(full_text_url, title, authors, publication_date)
        
        print("---")
    except requests.RequestException as e:
        print(f"Failed to retrieve article details: {e}")

def download_paper(pmc_url, paper_title, authors, publication_date):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(pmc_url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
        return
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = soup.find('a', {'class': 'pdf-link'})
        
        if not pdf_link:
            pdf_link = soup.find('a', {'href': lambda x: x and x.endswith('.pdf')})
        
        if pdf_link:
            pdf_url = pdf_link['href']
            if not pdf_url.startswith("http"):
                pdf_url = f"https://www.ncbi.nlm.nih.gov{pdf_url}"
            
            try:
                pdf_response = requests.get(pdf_url, headers=headers)
                pdf_response.raise_for_status()
                
                pdf_path = f"{paper_title}.pdf"
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_response.content)
                print(f"Downloaded: {pdf_path}")
                
                sections = extract_text_from_pdf(pdf_path)
                sections["Title"] = paper_title
                sections["Authors"] = authors
                sections["Publication Date"] = publication_date
                print(sections)
            except requests.RequestException as e:
                print(f"Failed to download the PDF: {e}")
        else:
            print(f"No PDF found for: {paper_title}")
    except Exception as e:
        print(f"Error processing PDF link: {e}")

def extract_text_from_pdf(pdf_path):
    text_by_page = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_by_page.append(text)
    
    sections = find_sections(text_by_page)
    return sections

# Dictionary intialization
def find_sections(text_by_page):
    sections = {
        "Abstract": "",
        "Introduction": "",
        "Methods": "",
        "Results": "",
        "Discussion": "",
        "Conclusion": ""
    }
    current_section = None
    section_text = ""

    # Updated regex
    section_patterns = {
        "Abstract": re.compile(r'^\s*ABSTRACT\s*$', re.IGNORECASE),
        "Introduction": re.compile(r'^\s*INTRODUCTION\s*$', re.IGNORECASE),
        "Methods": re.compile(r'^\s*METHODS\s*$', re.IGNORECASE),
        "Results": re.compile(r'^\s*RESULTS\s*$', re.IGNORECASE),
        "Discussion": re.compile(r'^\s*DISCUSSION\s*$', re.IGNORECASE),
        "Conclusion": re.compile(r'^\s*CONCLUSION\s*$', re.IGNORECASE),
    }

    for page_number, page in enumerate(text_by_page):
        lines = page.split('\n')
        
        for line_number, line in enumerate(lines):
            for section, pattern in section_patterns.items():
                if pattern.match(line.strip()):
                    if current_section:
                        sections[current_section] = section_text.strip()
                    current_section = section
                    section_text = ""
                    print(f"Found section {current_section} on page {page_number + 1}, line {line_number + 1}")
                    break
            
            if current_section:
                section_text += line + " "
    
    if current_section:
        sections[current_section] = section_text.strip()
    
    return sections

if __name__ == "__main__":
    scrape_page(current_url)
