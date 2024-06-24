import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    text_by_page = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1, y_tolerance=1)
            if text:
                text_by_page.append(text)
    return text_by_page

# Path to the PDF file
pdf_path = "Cross-Sectional Associations between Prenatal Per- and Poly-Fluoroalkyl Substances and Bioactive Lipids in Three Environmental Influences on Child Health Outcomes (ECHO) Cohorts.pdf"

# Extract text from PDF
text_by_page = extract_text_from_pdf(pdf_path)

# Print out the first few lines of each page to inspect the structure
for page_number, page in enumerate(text_by_page):
    print(f"--- Page {page_number + 1} ---")
    lines = page.split('\n')
    for line_number, line in enumerate(lines[:10]):  # Print the first 10 lines of each page
        print(f"{line_number + 1}: {line}")
    print("\n" + "="*80 + "\n")

def find_sections(text_by_page):
    sections = {}
    current_section = None
    section_text = ""

    #regex to capture variations in section headers
    section_pattern = re.compile(r'^\s*(ABSTRACT|METHODS?|INTRODUCTION|RESULTS?|DISCUSSION|CONCLUSION|REFERENCES?|ACKNOWLEDGEMENTS?|FUNDING|OBJECTIVE|BACKGROUND)\s*$', re.IGNORECASE)
    section_pattern_with_colon = re.compile(r'^\s*(ABSTRACT|METHODS?|INTRODUCTION|RESULTS?|DISCUSSION|CONCLUSION|REFERENCES?|ACKNOWLEDGEMENTS?|FUNDING|OBJECTIVE|BACKGROUND):\s*', re.IGNORECASE)

    for page_number, page in enumerate(text_by_page):
        lines = page.split('\n')
        
        for line_number, line in enumerate(lines):
            #print(line)
            
            header_match = section_pattern_with_colon.match(line.strip())
            if header_match:
                print("Identified section")
                if current_section:
                    sections[current_section] = section_text.strip()
                current_section = header_match.group(1).upper()
                section_text = ""
                print(f"Found section {current_section} on page {page_number + 1}, line {line_number + 1}")
            if current_section:
                section_text += line + " "
    
    if current_section:
        sections[current_section] = section_text.strip()
    
    return sections

# Find and extract sections
sections = find_sections(text_by_page)
