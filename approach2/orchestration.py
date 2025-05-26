import io
import mimetypes
import numpy as np
from PIL import Image
from document_ai.document_ai_client import process_document as process_document_ai
from ocr_utils.utils import merge_nearby_boxes, merge_overlapping_boxes, merge_row_boxes, merge_vertical_overlap_boxes, extend_to_full_width, annotate_image
import cv2
from common.document_ai_processor import process_and_annotate
from approach2.margin_crop import margin_crop_images
from approach1.orchestration import crop_questions

# Process images via Sobel margin crop followed by Document AI
def process_images(uploaded_files):
    """
    Process each uploaded file by performing margin crop and then Document AI.
    Returns list of dicts with keys: file_name, disp, annotated, annotated_bytes, boxes_json, logs, result_json, processing_time.
    """
    results = []
    margin_crops = margin_crop_images(uploaded_files)
    for file_name, (crop_img, margin_x, margin_sum) in margin_crops.items():
        # Encode cropped image to PNG bytes
        _, buf = cv2.imencode('.png', cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        file_bytes = buf.tobytes()
        mime_type = 'image/png'
        # Use common Document AI processor
        doc_result = process_and_annotate(file_bytes, mime_type)
        # Prepend margin crop log
        doc_result['logs'] = [f"🗜️ Margin crop for {file_name} @ x={margin_x}, sum={margin_sum}"] + doc_result.get('logs', [])
        doc_result['file_name'] = file_name
        results.append(doc_result)
    return results

# Crop each merged box region with half-gap vertical padding
def crop_boxes(image, boxes_json):
    # Determine image height for boundary checks
    height = image.shape[0]
    # Build regions list from boxes_json
    regions = []
    for box in boxes_json:
        idx = box["id"]
        x, y, w, h = box["bbox"]
        regions.append({
            "id": idx,
            "x_min": x,
            "x_max": x + w,
            "y_min": y,
            "y_max": y + h
        })
    # Sort regions vertically
    regions.sort(key=lambda r: r["y_min"])
    crops = {}
    # Crop with half-gap between boxes
    for i, r in enumerate(regions):
        y_min, y_max = r["y_min"], r["y_max"]
        # Upper padding (first box extends to top)
        if i > 0:
            prev_y_max = regions[i - 1]["y_max"]
            half_gap = (y_min - prev_y_max) / 2.0
            y_min_adj = int(max(0, y_min - half_gap))
        else:
            y_min_adj = 0
        # Lower padding from midpoint with next box
        if i < len(regions) - 1:
            next_y_min = regions[i + 1]["y_min"]
            half_gap2 = (next_y_min - y_max) / 2.0
            y_max_adj = int(min(height, y_max + half_gap2))
        else:
            y_max_adj = height
        x_min, x_max = r["x_min"], r["x_max"]
        # Ensure valid crop region
        if x_min < x_max and y_min_adj < y_max_adj:
            crops[r["id"]] = image[y_min_adj:y_max_adj, x_min:x_max]
    return crops
