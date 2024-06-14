import requests
from bs4 import BeautifulSoup
import PyPDF2

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
    pmc_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7961173/"
    paper_title = "Paper3"
    download_paper(pmc_url, paper_title)
