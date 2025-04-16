import os
import re
import fitz
import pdfplumber
import pytesseract
import pandas as pd
import unicodedata
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List
from pytesseract import Output
from typing import Dict, List
from llama_index.core import Document

class PDFProcessor:
    ''' Helps to handle the PDF info extraction
    reading text, tables, or scanned information'''
    
    def __init__(self, ocr_lang='fra'):
        self.ocr_lang = ocr_lang

    def process_pdf(self, pdf_path: str) -> Dict[int, Dict]:
    
        ''' Processes PDF и extracts text and tables from all pages'''
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Didn't find PDF file {pdf_path}")

        result = {}
        doc = fitz.open(pdf_path)

        try:
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_idx in range(len(doc)):
                    try:
                        result[page_idx] = self.process_page(
                            doc, plumber_pdf, page_idx
                        )
                    except Exception as e:
                        print(f"Error on page {page_idx + 1}: {e}")
                        result[page_idx] = {
                            "page_number": page_idx + 1,
                            "text": "",
                            "tables": [],
                            "ocr_text": "",
                            "error": str(e)
                        }
        finally:
            doc.close()

        return result

    def process_page(self, doc: fitz.Document, plumber_pdf: pdfplumber.PDF, 
                     page_idx: int) -> Dict:
        ''' Processes one page of PDF including text, tables and OCR (if needed)'''
        
        page_data = {
            "page_number": page_idx + 1,
            "text": "",
            "tables": [],
            "ocr_text": ""
        }

        fitz_page = doc[page_idx]
        plumber_page = plumber_pdf.pages[page_idx]

        try:
            text = fitz_page.get_text("text") or ""
            page_data["text"] = text
        except Exception as e:
            print(f"Text extraction failed on page {page_idx + 1}: {e}")

        try:
            tables = []
            plumber_tables = plumber_page.extract_tables()
            for table in plumber_tables:
                if table:
                    df = pd.DataFrame(table)
                    tables.append({
                        "data": df.values.tolist(),
                        "dataframe": df,
                        "source": "pdfplumber"
                    })
            page_data["tables"] = tables
            
        except Exception as e:
            print(f"PDFPLUMBER - Error on page {page_idx + 1}: {e}")

        if len(page_data["text"].strip()) < 70 and not page_data["tables"]:
            try:
                page_data["ocr_text"] = self.apply_ocr(fitz_page)
                if page_data["ocr_text"].strip():
                    page_data["text"] = page_data["ocr_text"]
                    ocr_table_df = self.extract_table_from_image(fitz_page)
                    if ocr_table_df is not None:
                        page_data["tables"].append({
                            "data": ocr_table_df.values.tolist(),
                            "dataframe": ocr_table_df,
                            "source": "ocr_image"
                        })
            except Exception as e:
                print(f"OCR - error on page {page_idx + 1}: {e}")

        return page_data

    def extract_table_from_image(self, page: fitz.Page) -> pd.DataFrame:
        ''' Extracts table from image if possible. Is used when 
        other table extraction method didn't work '''
        
        
        try:
            dpi = 200
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Use pytesseract to extract data as a DataFrame
            data = pytesseract.image_to_data(img_cv, output_type=Output.DATAFRAME, lang=self.ocr_lang)
            data = data[data.text.notnull()]

            if data.empty:
                return None

            lines = data.groupby("line_num")["text"].apply(lambda x: [w for w in x if w.strip() != ""]).tolist()
            lines = [line for line in lines if line]
            
            if not lines:
                return None

            df = pd.DataFrame(lines)
            return df
        
        except Exception as e:
            print(f"extract_table_from_image: Failed to extract table: {e}")
            return None

    def apply_ocr(self, page: fitz.Page) -> str:
        ''' Provides a text from an image.  Is used when
        page.get_text() gave None or some irrelevant type of data'''
        
        try:
            dpi = 200
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # convert to OpenCV format and enhance image contrast/brightness
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_cv = cv2.convertScaleAbs(img_cv, alpha=1.3, beta=20)
            enhanced_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

            # OCR configuration: LSTM 
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(
                enhanced_img,
                lang=self.ocr_lang,
                config=custom_config
            )
            return text.strip()
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
        

    def clean_document(self, doc: Dict) -> Dict:
        if "text" in doc:
            doc["text"] = self.clean_text(doc["text"])
            
        for table in doc.get("tables", []):
            for row in table.get("data", []):
                for i, cell in enumerate(row):
                    row[i] = self.clean_text(cell)
        return doc

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\(cid:\d+\)', '', text)
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬃ", "ffi").replace("ﬄ", "ffl")
        text = unicodedata.normalize("NFKC", text)  # normalize Unicode
        text = re.sub(r'\s+', ' ', text) #  collapse multiple spaces
        text = re.sub(r'(\n\s*){2,}', '\n\n', text.strip())
        return text.strip()
    
    
def prepare_llama_documents(results: Dict[int, Dict]) -> List[Document]:
    '''Converts a dict of page results into llama_index Document object
    with text, tables in Markdown format and metadata'''
    
    documents = []
    for i, data in results.items():
        page_num = data.get("page_number", i)
        text = data.get("text", "").strip()
        if text:
            documents.append(Document(text=text, metadata={"page": page_num, "type": "text"}))
        for j, table in enumerate(data.get("tables", [])):
            df_data = table.get("data", [])
            if df_data:
                header = df_data[0]
                rows = df_data[1:]
                markdown_table = "| " + " | ".join(map(str, header)) + " |\n"
                markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
                for row in rows:
                    markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
                documents.append(Document(text=markdown_table, metadata={"page": page_num, "type": "table", "table_number": j + 1}))
    return documents
