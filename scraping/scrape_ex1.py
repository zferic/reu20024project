import requests
from bs4 import BeautifulSoup
import time

# Define the base URL and search query URL
base_url = "https://pubmed.ncbi.nlm.nih.gov"
search_url = "https://pubmed.ncbi.nlm.nih.gov/?term=%28p42es017198%5BGrant+Number%5D%29+OR+%28p42+es017198%5BGrant+Number%5D%29&sort=date"

# Function to get the list of paper URLs from the search page
def get_paper_links(search_url):
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', class_='docsum-title')
    paper_links = [base_url + link['href'] for link in links]
    return paper_links

# Function to get the free full-text link for a given paper URL
def get_free_full_text_link(paper_url):
    response = requests.get(paper_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    #TODO this is not returning the link item 
    free_text_link = soup.find('a', {'class': 'link-item pmc-item'})


    print("free_text_link",free_text_link)
    if free_text_link:
        return free_text_link['href']
    return None

# Function to download the full-text PDF from the PMC page
def download_paper(pmc_url, paper_title):
    response = requests.get(pmc_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_link = soup.find('a', {'class': 'pdf-link'})
    
    if pdf_link:
        pdf_url = pdf_link['href']
        pdf_response = requests.get(pdf_url)
        with open(f"{paper_title}.pdf", 'wb') as f:
            f.write(pdf_response.content)
        print(f"Downloaded: {paper_title}.pdf")
    else:
        print(f"No PDF found for: {paper_title}")

# Main script
def main():
    paper_links = get_paper_links(search_url)
    for link in paper_links:
        time.sleep(2)  # To avoid overloading the server with requests
        free_full_text_link = get_free_full_text_link(link)
        if free_full_text_link:
            
            paper_title = link.split('/')[-2]  # Simple title extraction from URL
            
            pmc_url = f"https://www.ncbi.nlm.nih.gov{free_full_text_link}"
            download_paper(pmc_url, paper_title)
        else:
            print(f"No free full text available for: {link}")

if __name__ == "__main__":
    main()
