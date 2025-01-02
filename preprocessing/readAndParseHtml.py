import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import re
import json
import os
import time

base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "/?term=(p42es017198[Grant+Number])+OR+(p42+es017198[Grant+Number])&sort=date"
current_url = base_url + search_url

processed_pmids = set()

def scrape_page(url):
    print(f"Scraping URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paper_links = get_paper_links(soup)
        for paper_link in paper_links:
            extract_and_save_details(paper_link)
            time.sleep(1)  
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
        next_page_button = soup.find('a', class_='next-page')
        if next_page_button and 'disabled' not in next_page_button.get('class', []):
            next_page_href = next_page_button.get('href')
            next_page_url = urljoin(base_url, next_page_href)
        return next_page_url
    except Exception as e:
        print(f"Failed to get next page URL: {e}")
        return None

def extract_and_save_details(paper_url):
    try:
        pmid = extract_pmid_from_url(paper_url)
        if not pmid:
            print(f"Could not extract PMID from URL: {paper_url}. Skipping.")
            return
        
        if pmid in processed_pmids:
            print(f"Duplicate PMID found. Skipping paper with PMID: {pmid}")
            return
        processed_pmids.add(pmid)
        
        response = requests.get(paper_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_tag = soup.find('h1', class_='heading-title')
        title = title_tag.text.strip() if title_tag else 'N/A'
        
        author_tags = soup.find_all('a', class_='full-name')
        authors = [author.text.strip() for author in author_tags]
        
        date_tag = soup.find('span', class_='cit')
        publication_date = date_tag.text.strip() if date_tag else 'N/A'
        
        paper_data = {
            "pmid": pmid,
            "title": title,
            "url": paper_url,
            "authors": authors,
            "publication_date": publication_date,
            "sections": {},
            "tables": {}
        }
        
        full_text_links = soup.select('a.link-item.pmc, a.link-item.dialog-focus')
        for full_text_link in full_text_links:
            full_text_url = full_text_link['href']
            full_text_url = urljoin(base_url, full_text_url)
            print(f"  Full Text URL: {full_text_url}")
            parse_full_text(full_text_url, paper_data)
            time.sleep(1)  

        save_paper_to_files(paper_data)
        
        print(f"Processed and saved paper: {title} (PMID: {pmid})")
        print("---")
        
    except requests.RequestException as e:
        print(f"Failed to retrieve article details: {e}")
    except Exception as e:
        print(f"An error occurred while extracting details: {e}")

def extract_pmid_from_url(url):
    """
    Extracts the PMID from the PubMed URL.
    Example URL: https://pubmed.ncbi.nlm.nih.gov/12345678/
    """
    try:
        match = re.search(r'/(\d+)/?$', url)
        if match:
            return match.group(1)
        else:
            return None
    except Exception as e:
        print(f"Error extracting PMID from URL {url}: {e}")
        return None

def parse_full_text(full_text_url, paper_data):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(full_text_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        sections = extract_sections_from_html(soup)
        paper_data["sections"] = sections
        
        tables = extract_tables_from_html(soup)
        paper_data["tables"] = tables
        
    except requests.RequestException as e:
        print(f"Failed to retrieve the full text page: {e}")
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
        current_section = None
        section_text = ""

        for header in headers:
            header_text = header.get_text().strip()
            header_text_lower = header_text.lower()

            matched_section = None
            for section, pattern in section_patterns.items():
                if pattern.search(header_text_lower):
                    matched_section = section
                    break

            if matched_section:
                if current_section and section_text:
                    sections[current_section] = section_text.strip()
                current_section = matched_section
                section_text = ""
                print(f"Detected section: {matched_section}")
                continue 

            if current_section:
                for sibling in header.find_next_siblings():
                    if sibling.name and re.match('^h[1-6]$', sibling.name):
                        break
                    if hasattr(sibling, 'get_text'):
                        section_text += sibling.get_text(separator=' ', strip=True) + " "
                sections[current_section] = section_text.strip()
                section_text = ""
                current_section = None  

        if current_section and section_text:
            sections[current_section] = section_text.strip()

    except Exception as e:
        print(f"An error occurred while extracting sections from HTML: {e}")

    return sections

def extract_tables_from_html(soup):
    tables = {}
    try:
        table_links = soup.find_all('a', class_='usa-link', href=re.compile(r'^#T\d+$'))
        for link in table_links:
            table_id = link['href'].lstrip('#')
            table_number = link.get_text().strip()
            table_element = soup.find(id=table_id)
            if table_element:
                table_data = parse_html_table(table_element)
                tables[table_number] = table_data
    except Exception as e:
        print(f"An error occurred while extracting tables: {e}")
    return tables

def parse_html_table(table_element):
    table = []
    try:
        headers = []
        thead = table_element.find('thead')
        if thead:
            headers = [th.get_text().strip() for th in thead.find_all('th')]
        else:
            first_row = table_element.find('tr')
            if first_row:
                headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
        
        for row in table_element.find_all('tr')[1:]:  
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            row_data = {}
            for idx, cell in enumerate(cells):
                header = headers[idx] if idx < len(headers) else f"Column {idx+1}"
                row_data[header] = cell.get_text().strip()
            table.append(row_data)
    except Exception as e:
        print(f"An error occurred while parsing a table: {e}")
    return table

def save_paper_to_files(paper_data):
    try:
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", paper_data['title'])
        os.makedirs('testpapers', exist_ok=True)

        txt_filename = f"{safe_title}.txt"
        txt_path = os.path.join('testpapers', txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(f"PMID: {paper_data['pmid']}\n")
            txt_file.write(f"Title: {paper_data['title']}\n")
            txt_file.write(f"URL: {paper_data['url']}\n")
            txt_file.write(f"Authors: {', '.join(paper_data['authors'])}\n")
            txt_file.write(f"Publication Date: {paper_data['publication_date']}\n\n")
            for section, content in paper_data['sections'].items():
                txt_file.write(f"### {section} ###\n")
                txt_file.write(content + "\n\n")
        
        print(f"Saved metadata and sections to {txt_path}")
        
        if paper_data['tables']:
            json_filename = f"{safe_title}_tables.json"
            json_path = os.path.join('testpapers', json_filename)
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(paper_data['tables'], json_file, ensure_ascii=False, indent=4)
            print(f"Saved tables to {json_path}")
        else:
            print("No tables found to save.")
        
    except Exception as e:
        print(f"An error occurred while saving the paper to files: {e}")

if __name__ == "__main__":
    scrape_page(current_url)
