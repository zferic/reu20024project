import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
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
        paper_links = get_paper_links(soup)
        for paper_link in paper_links:
            extract_and_print_details(paper_link)
        next_page_url = get_next_page_url(soup, url)
        if next_page_url:
            scrape_page(next_page_url)
    except requests.RequestException as e:
        print(f"Failed to scrape the page: {e}")

def get_paper_links(soup):
    try:
        links = soup.find_all('a', class_='docsum-title')
        paper_links = [urljoin(base_url, link['href']) for link in links]
        return paper_links
    except Exception as e:
        print(f"Failed to get paper links: {e}")
        return []

def get_next_page_url(soup, current_url):
    try:
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
    except Exception as e:
        print(f"Failed to get next page URL: {e}")
        return None

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
            parse_full_text(full_text_url, title, authors, publication_date)
        
        print("---")
    except requests.RequestException as e:
        print(f"Failed to retrieve article details: {e}")
    except Exception as e:
        print(f"An error occurred while extracting details: {e}")

def parse_full_text(full_text_url, paper_title, authors, publication_date):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(full_text_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        sections = extract_sections_from_html(soup)
        save_sections_to_file(paper_title, authors, publication_date, sections)
    except requests.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
    except Exception as e:
        print(f"An error occurred while parsing the full text: {e}")

def extract_sections_from_html(soup):
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

    section_patterns = {
        "Abstract": re.compile(r'\babstract\b', re.IGNORECASE),
        "Introduction": re.compile(r'\bintroduction\b', re.IGNORECASE),
        "Methods": re.compile(r'\b(?:materials\s*and\s*methods?|methods?|methodology)\b', re.IGNORECASE),
        "Results": re.compile(r'\bresults?\b', re.IGNORECASE),
        "Discussion": re.compile(r'\bdiscussions?\b', re.IGNORECASE),
        "Conclusion": re.compile(r'\b(?:conclusions?|in\s*conclusion)\b', re.IGNORECASE),
    }

    try:
        headers = soup.find_all(re.compile('^h[1-6]$'))
        for header in headers:
            header_text = header.get_text().strip()
            for section, pattern in section_patterns.items():
                if pattern.search(header_text):
                    if current_section:
                        sections[current_section] = section_text.strip()
                    current_section = section
                    section_text = ""
                    break

            if current_section:
                next_sibling = header.find_next_sibling()
                while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if hasattr(next_sibling, 'get_text'):
                        section_text += next_sibling.get_text(separator=' ', strip=True) + "\n"
                    if next_sibling.name and re.match('^h[1-6]$', next_sibling.name):
                        section_text += f"\n{subheader.get_text().strip()}\n"
                    next_sibling = next_sibling.find_next_sibling()

        if current_section:
            sections[current_section] = section_text.strip()

    except Exception as e:
        print(f"An error occurred while extracting sections from HTML: {e}")

    return sections

def save_sections_to_file(paper_title, authors, publication_date, sections):
    try:
        paper_title = re.sub(r'[\\/*?:"<>|]', "_", paper_title)
        directory = r"C:\Users\tiahi\NSF REU\tokenizing\updated_txts"
        os.makedirs(directory, exist_ok=True)
        file_name = os.path.join(directory, f"{paper_title}_sections.txt")
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"Title: {paper_title}\n")
            f.write(f"Authors: {authors}\n")
            f.write(f"Publication Date: {publication_date}\n\n")
            for section_name, section_text in sections.items():
                f.write(f"### {section_name} ###\n\n")
                f.write(section_text.strip() + "\n\n")
        print(f"Saved sections to {file_name}")
    except Exception as e:
        print(f"An error occurred while saving sections to file: {e}")


if __name__ == "__main__":
    scrape_page(current_url)
