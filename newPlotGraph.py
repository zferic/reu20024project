import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import re
import os
import csv
import matplotlib.pyplot as plt

# Base URL and initial search URL
base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "/?term=contaminant+fate+and+transport&sort=date&page=1"
current_url = base_url + search_url

# Directory for saving the text files
save_directory = r"C:\Users\musak\OneDrive\REU Folder\Datascraping\downloaded_text"

def scrape_pages(url):
    while url:
        print("Scraping URL: " + url)
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            paper_links = get_paper_links(soup)
            for paper_link in paper_links:
                extract_and_update_csv(paper_link)
            url = get_next_page_url(soup, url)
        except requests.RequestException as e:
            print(f"Failed to scrape the page: {e}")
            break

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

def extract_and_update_csv(paper_url):
    try:
        response = requests.get(paper_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', class_='heading-title')
        title = title_tag.text.strip() if title_tag else 'N/A'
        
        sections_present = extract_sections_from_html(soup)
        
        # Update CSV with sections presence
        update_csv(title, sections_present)
        
    except requests.RequestException as e:
        print(f"Failed to retrieve article details: {e}")
    except Exception as e:
        print(f"An error occurred while extracting details: {e}")

def extract_sections_from_html(soup):
    sections_present = {
        "Abstract": False,
        "Introduction": False,
        "Methods": False,
        "Results": False,
        "Discussion": False,
        "Conclusion": False
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
        for header in headers:
            header_text = header.get_text().strip()
            for section, pattern in section_patterns.items():
                if pattern.search(header_text):
                    sections_present[section] = True

        return sections_present

    except Exception as e:
        print(f"An error occurred while extracting sections from HTML: {e}")
        return sections_present

def update_csv(paper_title, sections_present):
    csv_file = os.path.join(save_directory, "sections_presence.csv")
    fieldnames = ['Paper Title', 'Abstract', 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion']
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            'Paper Title': paper_title,
            'Abstract': sections_present.get('Abstract', False),
            'Introduction': sections_present.get('Introduction', False),
            'Methods': sections_present.get('Methods', False),
            'Results': sections_present.get('Results', False),
            'Discussion': sections_present.get('Discussion', False),
            'Conclusion': sections_present.get('Conclusion', False),
        })

        print(f"Updated CSV file with sections presence for {paper_title}")

    # Update the graph after each CSV update
    section_counts = read_csv_data(csv_file)
    plot_bar_graph(section_counts)

def read_csv_data(csv_file):
    section_counts = {
        'Abstract': 0,
        'Introduction': 0,
        'Methods': 0,
        'Results': 0,
        'Discussion': 0,
        'Conclusion': 0
    }

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            section_counts['Abstract'] += int(row['Abstract'] == 'True')
            section_counts['Introduction'] += int(row['Introduction'] == 'True')
            section_counts['Methods'] += int(row['Methods'] == 'True')
            section_counts['Results'] += int(row['Results'] == 'True')
            section_counts['Discussion'] += int(row['Discussion'] == 'True')
            section_counts['Conclusion'] += int(row['Conclusion'] == 'True')

    return section_counts

def plot_bar_graph(section_counts):
    sections = list(section_counts.keys())
    counts = list(section_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(sections, counts, color='skyblue')
    plt.xlabel('Sections')
    plt.ylabel('Count')
    plt.title('Presence of Sections in Papers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, 'sections_presence_graph.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    # Create or truncate the CSV file
    csv_file = os.path.join(save_directory, "sections_presence.csv")
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Paper Title', 'Abstract', 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    # Scrape pages and update CSV with sections presence
    scrape_pages(current_url)

    # Final read CSV data and plot graph
    section_counts = read_csv_data(csv_file)
    plot_bar_graph(section_counts)
