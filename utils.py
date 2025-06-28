import pdfplumber

def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def read_txt(file):
    return file.read().decode('utf-8')
