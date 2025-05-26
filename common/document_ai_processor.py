import io
import mimetypes
import numpy as np
from PIL import Image
import base64
import time
import requests
import os
import google.auth
import google.auth.transport.requests
from dotenv import load_dotenv
load_dotenv()
from ocr_utils.utils import merge_nearby_boxes, merge_overlapping_boxes, merge_row_boxes, merge_vertical_overlap_boxes, extend_to_full_width, annotate_image

# Configuration for Google Document AI
PROJECT_ID = "629159515213"
LOCATION = "us"
PROCESSOR_ID = "425f4be70b86af78"
ENDPOINT_URL = f"https://{LOCATION}-documentai.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}:process"

def get_access_token():
    """Get access token using Application Default Credentials"""
    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token

def process_document(file_content, mime_type):
    """Process a document using Google Document AI REST API"""
    try:
        access_token = get_access_token()
        if not access_token:
            return {"error": "Could not retrieve access token"}

        # Encode document content
        encoded_content = base64.b64encode(file_content).decode("utf-8")
        payload = {
            "rawDocument": {
                "content": encoded_content,
                "mimeType": mime_type
            }
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        start_time = time.time()
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        processing_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            return {"result": result, "processing_time": processing_time}
        else:
            return {"error": f"API Error {response.status_code}: {response.text}", "processing_time": processing_time}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def process_and_annotate(file_bytes, mime_type):
    """
    Process a document using Google Document AI, merge boxes, and annotate the image.
    Returns a dict with keys: disp, annotated, annotated_bytes, boxes_json, logs, result_json, processing_time.
    """
    # Call Document AI
    response = process_document(file_bytes, mime_type)
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