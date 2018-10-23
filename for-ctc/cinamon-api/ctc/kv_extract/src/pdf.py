import difflib
import os

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTTextLineHorizontal
from pdfminer.converter import PDFPageAggregator
from PyPDF2 import PdfFileWriter, PdfFileReader

SPECIAL_CHARS = ['.', '/', '-']

class PDFAnalyzer(object):
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.pages = []
        self._layouts = []
        self.header_lines = []
        self.footer_lines = []
        self.extract_layouts()
        self.extract_textline()
        self.header_size = self.detect_header_size()
        self.footer_size = self.detect_footer_size()

    def extract_layouts(self):
        f = open(self.pdf_file, 'rb')
        self.parser = PDFParser(f)        
        self.document = PDFDocument(self.parser)
        if not self.document.is_extractable:
            raise PDFTextExtractionNotAllowed
        self.rsrcmgr = PDFResourceManager()
        self.laparams = LAParams()
        self.device = PDFPageAggregator(self.rsrcmgr, laparams=self.laparams)
        self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)

        for page in PDFPage.create_pages(self.document):
            self.interpreter.process_page(page)
            layout = self.device.get_result()
            self._layouts.append(layout)
        f.close()

    def extract_textline(self):
        for layout in self._layouts:
            lines = []
            for obj in layout._objs:
                if isinstance(obj, LTTextLineHorizontal):
                    line = _make_line_obj(obj)
                    lines.append(line)
                if isinstance(obj, LTTextBoxHorizontal):
                    for o in obj:
                        if isinstance(o, LTTextLineHorizontal):
                            line = _make_line_obj(o)
                            lines.append(line)
            sorted_lines = sorted(lines, key = lambda item: (item['upperLeft'][1], item['upperLeft'][0]), reverse=True)
            self.pages.append(sorted_lines)

    def detect_header_size(self):
        if len(self.pages) < 2:
            return 0

        page1 = self.pages[0]
        page2 = self.pages[1]
        limit_size = self._layouts[0].y1 * 3.0 / 5 # limit 1/3 upper size of page 

        for idx, line in enumerate(page2):
            if idx < len(page1):
                base_line = page1[idx]
            else:
                break
            if base_line['upperLeft'][1] < limit_size:
                break
            if line['text'] == base_line['text'] and line['upperLeft'][0] == base_line['upperLeft'][0]:
                self.header_lines.append(base_line)

        if not self.header_lines:
            return 0

        return self.header_lines[-1]['lowerLeft'][1] # last 

    def detect_footer_size(self):
        if len(self.pages) < 2:
            return 0

        page1 = self.pages[0]
        page2 = self.pages[1]
        limit_size = self._layouts[0].y1 * 1.0 / 5 # limit 1/3 lower size of page

        for idx, line in enumerate(page2[::-1], start=1): # start from bottom line
            if idx < len(page1):
                base_line = page1[0-idx]
            else:
                break
            if base_line['lowerLeft'][1] > limit_size:
                break
            if line['upperLeft'][0] == base_line['upperLeft'][0]:
                if line['text'] == base_line['text']: # same text
                    self.footer_lines.append(base_line)
                else:
                    if line['text'].isnumeric() and base_line['text'].isnumeric():
                        self.footer_lines.append(base_line)
                        continue
                    if len(line['text']) >= 3 and len(line['text']) == len(base_line['text']):
                        m = difflib.SequenceMatcher(None, line['text'], base_line['text'])
                        if m.ratio() > 0.6:
                            self.footer_lines.append(base_line)
        
        if not self.footer_lines:
            return 0

        return self.footer_lines[-1]['upperLeft'][1]


def _make_line_obj(pdf_text_line): # make compatible with PyPDF2
    text = pdf_text_line.get_text().strip()
    upper_left = pdf_text_line.x0, pdf_text_line.y1
    upper_right = pdf_text_line.x1, pdf_text_line.y1
    lower_left = pdf_text_line.x0, pdf_text_line.y0
    lower_right = pdf_text_line.x1, pdf_text_line.y0
    return {
        "text": text,
        "upperLeft": upper_left,
        "upperRight": upper_right,
        "lowerLeft": lower_left,
        "lowerRight": lower_right
    }


def remove_header_footer(input_file, output_file):
    analyzer = PDFAnalyzer(input_file)
    if analyzer.header_size == 0 and analyzer.footer_size == 0:
        return input_file

    pdf_reader = PdfFileReader(input_file, strict=False)
    output = PdfFileWriter()

    num_pages = pdf_reader.getNumPages()
    for i in range(num_pages):
        page = pdf_reader.getPage(i)
        if analyzer.header_size != 0:
            page.mediaBox.upperLeft = page.mediaBox.upperLeft[0], analyzer.header_size - 4 # un-compatible between pdfminer object and pypdf2 crop
            page.mediaBox.upperRight = page.mediaBox.upperRight[0], analyzer.header_size - 4
        if analyzer.footer_size != 0:
            page.mediaBox.lowerRight = page.mediaBox.lowerRight[0], analyzer.footer_size
            page.mediaBox.lowerLeft = page.mediaBox.lowerLeft[0], analyzer.footer_size
        output.addPage(page)
    with open(output_file, 'wb') as f:
        output.write(f)
    return output_file
