#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexiForge OCR Handler
=====================
Extracts text from PDF pages using PyMuPDF (fitz) for rendering and PaddleOCR for text recognition.
Implements a dikey sütun sıralama (vertical column sorting) heuristic to preserve dictionary layouts.
"""

import os
import sys
import numpy as np

# Cache PaddleOCR instances to avoid loading models repeatedly / Model yükleme maliyetini önlemek için cache'leriz
_ocr_instances = {}


def get_ocr_instance(lang_code, use_gpu=False):
    """
    Returns or initializes a cached PaddleOCR instance for the target language.
    """
    # Map ISO language code to PaddleOCR language codes
    # ISO dil kodlarını PaddleOCR dil kodlarına eşleriz
    lang_map = {
        "DE": "german",
        "TR": "turkish",
        "EN": "english",
        "FR": "french",
        "RU": "russian",
        "IT": "italian",
        "ES": "spanish",
        "PT": "portuguese",
        "LA": "latin",
        "RO": "romanian",
        "SR": "serbian",
        "MK": "macedonian"
    }
    ocr_lang = lang_map.get(lang_code.upper(), "english")
    
    key = (ocr_lang, use_gpu)
    if key in _ocr_instances:
        return _ocr_instances[key]
        
    try:
        from paddleocr import PaddleOCR
        # Disable logging to keep console clean unless debug is needed
        # Konsolu temiz tutmak için PaddleOCR loglarını kapatıyoruz
        ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang, use_gpu=use_gpu, show_log=False)
        _ocr_instances[key] = ocr
        return ocr
    except Exception as e:
        print(f"\033[1;31m[OCR] Failed to initialize PaddleOCR for language '{ocr_lang}': {e}\033[0m")
        return None


def extract_text_from_pdf_page(pdf_path, page_num, config, src_lang="DE"):
    """
    Renders a specific PDF page to an image and performs OCR with column heuristics.
    
    Parameters:
      - pdf_path: Absolute or relative path to PDF file.
      - page_num: 1-based index of the page.
      - config: Config dictionary containing ocr settings.
      - src_lang: Source language code.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("\033[1;31m[OCR] PyMuPDF (fitz) is not installed. Falling back to empty text.\033[0m")
        return ""

    ocr_cfg = config.get("ocr", {})
    enabled = ocr_cfg.get("enabled", True)
    if not enabled:
        return ""

    use_gpu = ocr_cfg.get("use_gpu", False)
    column_detection = ocr_cfg.get("column_detection", True)

    ocr = get_ocr_instance(src_lang, use_gpu=use_gpu)
    if not ocr:
        return ""

    try:
        # 1. Render page to image using PyMuPDF / Sayfayı PyMuPDF ile resme dönüştürme
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            print(f"\033[1;31m[OCR] Page number {page_num} out of bounds (1-{len(doc)})\033[0m")
            return ""
            
        page = doc[page_num - 1]
        
        # Render at 300 DPI for high quality OCR / Yüksek doğruluk için 300 DPI çözünürlükte render etme
        zoom = 300 / 72  # 72 is default PDF point size
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        
        # Convert pixmap to numpy array / Pixmap'i numpy array'e dönüştürme
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # If image has alpha channel, discard it
        if pix.n == 4:
            img_data = img_data[:, :, :3]
            
        width, height = pix.w, pix.h
        doc.close()
        
        # 2. Run PaddleOCR / PaddleOCR'ı çalıştırma
        result = ocr.ocr(img_data, cls=True)
        if not result or not result[0]:
            return ""
            
        boxes_and_text = result[0]
        
        # 3. Clean and parse layout / Sayfa düzenini ayrıştırma
        valid_boxes = []
        for box in boxes_and_text:
            coords = box[0]
            text, conf = box[1]
            
            # Find bounds
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Filter out headers and footers (Y axis threshold: top 5%, bottom 5%)
            # Sayfa numarası, üst bilgi ve alt bilgileri filtreleriz (Üst/Alt %5 dilim)
            if center_y < height * 0.05 or center_y > height * 0.95:
                continue
                
            valid_boxes.append({
                "text": text,
                "center_x": center_x,
                "center_y": center_y,
                "min_x": min_x,
                "max_x": max_x
            })
            
        if not valid_boxes:
            return ""
            
        # 4. Column Heuristics / Sütun Heuristiği
        if column_detection:
            # Midpoint X coordinate
            mid_x = width / 2
            
            left_column = []
            right_column = []
            
            for item in valid_boxes:
                # If center_x is on the left, it belongs to the left column
                if item["center_x"] < mid_x:
                    left_column.append(item)
                else:
                    right_column.append(item)
                    
            # Sort each column from top to bottom (Y axis)
            # Her sütunu yukarıdan aşağıya (Y eksenine göre) sıralarız
            left_column.sort(key=lambda x: x["center_y"])
            right_column.sort(key=lambda x: x["center_y"])
            
            # Merge columns
            sorted_items = left_column + right_column
        else:
            # Simple top-to-bottom reading order
            sorted_items = sorted(valid_boxes, key=lambda x: x["center_y"])
            
        text_lines = [item["text"] for item in sorted_items]
        return "\n".join(text_lines)
        
    except Exception as e:
        print(f"\033[1;31m[OCR] Exception occurred during OCR processing: {e}\033[0m")
        return ""
