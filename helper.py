import fitz
from PIL import Image
import numpy as np


def extract_chunk_image(pdf_path, page_num, bbox, output_path):

    doc = fitz.open(pdf_path)
    page = doc[page_num]

    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    w, h = pix.width, pix.height

    left = bbox["left"] * w
    top = bbox["top"] * h
    right = bbox["right"] * w
    bottom = bbox["bottom"] * h

    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)

    doc.close()