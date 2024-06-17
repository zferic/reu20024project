import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import PyPDF2

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
    
    # Full text function 
    full_text_url = soup.find('a', class_='link-item pmc')
    if full_text_url:
        full_text_url = full_text_url['href']
        full_text_url = urljoin(base_url, full_text_url)
        print(f"  Full Text URL: {full_text_url}")
        download_paper(full_text_url, title)
    
    print("---")

def download_paper(pmc_url, paper_title):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    response = requests.get(pmc_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    pdf_link = soup.find('a', {'class': 'pdf-link'})
    
    if not pdf_link:
        pdf_link = soup.find('a', {'href': lambda x: x and x.endswith('.pdf')})
    
    if pdf_link:
        pdf_url = pdf_link['href']
        if not pdf_url.startswith("http"):
            pdf_url = f"https://www.ncbi.nlm.nih.gov{pdf_url}"
        
        pdf_response = requests.get(pdf_url, headers=headers)
        
        if pdf_response.status_code == 200:
            pdf_path = f"{paper_title}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_response.content)
            print(f"Downloaded: {pdf_path}")
            
            extract_text_from_pdf(pdf_path)
        else:
            print(f"Failed to download the PDF. Status code: {pdf_response.status_code}")
    else:
        print(f"No PDF found for: {paper_title}")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdfFile:
        pdfReader = PyPDF2.PdfReader(pdfFile)
        text = ""
        for page in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[page]
            text += pageObj.extract_text()
        
        print(text)

if __name__ == "__main__":
    scrape_page(current_url)
