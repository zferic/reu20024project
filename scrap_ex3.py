import requests
from bs4 import BeautifulSoup
import time

base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "https://pubmed.ncbi.nlm.nih.gov/?term=(p42es017198[Grant+Number])+OR+(p42+es017198[Grant+Number])&sort=date"
current_url = search_url

def scrape_page(url):
    print(f"URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paper_links = get_paper_links(url)
    for link in paper_links:
        time.sleep(2)  
        print(f"Scraping article: {link}")
        extract_and_print_details(link)

def get_paper_links(search_url):
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', class_='docsum-title')
    paper_links = [base_url + link['href'] for link in links]
    return paper_links

def get_next_page_url(current_url):
    response = requests.get(current_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    next_page_tag = soup.find('button', class_='button-wrapper next-page-btn')
    if next_page_tag and 'href' in next_page_tag.attrs:
        next_page_url = base_url + next_page_tag['href']
        return next_page_url
    else:
        return None

def extract_and_print_details(paper_url):
    response = requests.get(paper_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title_tag = soup.find('h1', class_='heading-title')
    title = title_tag.text.strip() if title_tag else 'N/A'
    print(f"Article Title: {title}")
    author_tags = soup.find_all('a', class_='full-name')
    authors = ", ".join([author.text for author in author_tags])
    date_tag = soup.find('span', class_='cit')
    publication_date = date_tag.text.strip() if date_tag else 'N/A'
    print(f"  Authors: {authors}")
    print(f"  Publication Date: {publication_date}")
    print("---")

while current_url:
    scrape_page(current_url)
    current_url = get_next_page_url(current_url)
    if current_url:
        print(f"Moving on to the next page: {current_url}")
    else:
        print("No more pages to scrape!")