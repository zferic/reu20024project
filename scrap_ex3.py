import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "/?term=(p42es017198[Grant+Number])+OR+(p42+es017198[Grant+Number])&sort=date"
current_url = base_url + search_url

def scrape_page(url):
    print("Scraping URL: " + url)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    paper_links = get_paper_links(soup, url)
    for paper_link in paper_links:
        extract_and_print_details(paper_link)
    next_page_url = get_next_page_url(soup, url)
    if next_page_url:
        scrape_page(next_page_url)

def get_paper_links(soup, search_url):
    links = soup.find_all('a', class_='docsum-title')
    paper_links = [base_url + link['href'] for link in links]
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
    response = requests.get(paper_url)
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
    
    # Check if full text is available
    full_text_url = soup.find('a', class_='link-item pmc')
    if full_text_url:
        full_text_url = full_text_url['href']
        print(f"  Full Text URL: {full_text_url}")
    
    print("---")

scrape_page(current_url)