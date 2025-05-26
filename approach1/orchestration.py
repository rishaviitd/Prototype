import mimetypes
from common.document_ai_processor import process_and_annotate

# Process images using Document AI only
def process_images(uploaded_files, mode=None):
    """
    Process images using Google Document AI only.
    """
    results = []
    for file in uploaded_files:
        # Reset file pointer
        try:
            file.seek(0)
        except Exception:
            pass
        file_bytes = file.read()
        mime_type = file.type or mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        res = process_and_annotate(file_bytes, mime_type)
        res["file_name"] = file.name
        results.append(res)
    return results

# ─── Cropping ─────────────────────────────────────────────────────────────
def crop_questions(image, boxes_json, question_mapping):
    for q_id, box_ids in question_mapping.items():
        question_mapping[q_id] = [int(b) for b in box_ids]
    boxes_by_id = {box['id']: box['bbox'] for box in boxes_json}
    height, width = image.shape[:2]
    regions = []
    for q_id, box_ids in question_mapping.items():
        if not box_ids:
            continue
        valid_ids = [b for b in box_ids if b in boxes_by_id]
        if not valid_ids:
            continue
        x_min = min(boxes_by_id[b][0] for b in valid_ids)
        y_min = min(boxes_by_id[b][1] for b in valid_ids)
        x_max = max(boxes_by_id[b][0] + boxes_by_id[b][2] for b in valid_ids)
        y_max = max(boxes_by_id[b][1] + boxes_by_id[b][3] for b in valid_ids)
        regions.append({'q_id': q_id, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
    regions.sort(key=lambda r: r['y_min'])
    cropped = {}
    for idx, r in enumerate(regions):
        y0, y1 = r['y_min'], r['y_max']
        if idx > 0:
            prev_y1 = regions[idx-1]['y_max']
            half_gap = (y0 - prev_y1) / 2
            y0 = int(max(0, y0 - half_gap))
        if idx < len(regions)-1:
            next_y0 = regions[idx+1]['y_min']
            half_gap = (next_y0 - r['y_max']) / 2
            y1 = int(min(height, r['y_max'] + half_gap))
        x0, x1 = r['x_min'], r['x_max']
        if x0 < x1 and y0 < y1:
            cropped[r['q_id']] = image[y0:y1, x0:x1]
    return cropped 