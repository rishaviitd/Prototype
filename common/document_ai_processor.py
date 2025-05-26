import io
import mimetypes
import numpy as np
from PIL import Image
from document_ai.document_ai_client import process_document as process_doc_ai
from ocr_utils.utils import merge_nearby_boxes, merge_overlapping_boxes, merge_row_boxes, merge_vertical_overlap_boxes, extend_to_full_width, annotate_image

def process_and_annotate(file_bytes, mime_type):
    """
    Process a document using Google Document AI, merge boxes, and annotate the image.
    Returns a dict with keys: disp, annotated, annotated_bytes, boxes_json, logs, result_json, processing_time.
    """
    # Call Document AI
    response = process_doc_ai(file_bytes, mime_type)
    result_json = response.get("result", {})
    processing_time = response.get("processing_time", 0)
    document = result_json.get("document", {})
    pages = document.get("pages", [])
    if not pages:
        return {
            "disp": None,
            "annotated": None,
            "annotated_bytes": b"",
            "boxes_json": [],
            "logs": [f"⚠️ Document AI returned no pages"],
            "result_json": result_json,
            "processing_time": processing_time
        }
    # Only first page
    page = pages[0]
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    disp = np.array(img)
    height, width = disp.shape[:2]

    # Extract bounding boxes
    boxes = []
    if "blocks" in page:
        for block in page["blocks"]:
            if "boundingPoly" in block:
                verts = block["boundingPoly"].get("normalizedVertices", [])
            elif "layout" in block and "boundingPoly" in block["layout"]:
                verts = block["layout"]["boundingPoly"].get("normalizedVertices", [])
            else:
                continue
            xs = [int(v.get("x", 0) * width) for v in verts]
            ys = [int(v.get("y", 0) * height) for v in verts]
            if xs and ys:
                x_min, y_min = min(xs), min(ys)
                x_max, y_max = max(xs), max(ys)
                boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    # Merge boxes
    dist_thresh = np.median([h for (_, _, _, h) in boxes]) * 1.2 if boxes else 0
    boxes = merge_nearby_boxes(boxes, dist_thresh)
    boxes = merge_overlapping_boxes(boxes)
    boxes = merge_row_boxes(boxes, y_thresh_factor=0.5)
    boxes = merge_vertical_overlap_boxes(boxes)
    boxes = extend_to_full_width(boxes, width)

    # Annotate image
    annotated, annotated_bytes, boxes_json = annotate_image(disp, boxes)
    logs = [f"✅ Processed with Document AI ({mime_type})"]

    return {
        "disp": disp,
        "annotated": annotated,
        "annotated_bytes": annotated_bytes,
        "boxes_json": boxes_json,
        "logs": logs,
        "result_json": result_json,
        "processing_time": processing_time
    } 