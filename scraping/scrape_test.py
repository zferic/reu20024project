import requests
from bs4 import BeautifulSoup
import os
 
# URL from which pdfs to be downloaded
url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8295239/"
 
# Requests URL and get response object
response = requests.get(url)
 
# Parse text obtained
soup = BeautifulSoup(response.text, 'html.parser')
 
pdf_links = soup.select('.column-1 a, .column-3 a')

for link in pdf_links:
    pdf_url = link['href']
    pdf_response = requests.get(pdf_url)

    print(url)
    
    if 'column-1' in link.parent['class']:
        folder_name = 'Exmen'
    elif 'column-3' in link.parent['class']:
        folder_name = 'Correction'
    else:
        folder_name = 'unknown'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    with open(f"{folder_name}/{link.text}.pdf", 'wb') as f:
        f.write(pdf_response.content)