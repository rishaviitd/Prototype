import cv2
import numpy as np
import easyocr
import networkx as nx
from itertools import combinations
from math import sqrt

# ─── Box Merging Utilities ─────────────────────────────────────────────────
def merge_nearby_boxes(boxes, dist_thresh):
    def edge_dist(b1, b2):
        x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
        dx = max(x1 - (x2+w2), x2 - (x1+w1), 0)
        dy = max(y1 - (y2+h2), y2 - (y1+h1), 0)
        return sqrt(dx*dx + dy*dy)

    G = nx.Graph()
    G.add_nodes_from(range(len(boxes)))
    for i, j in combinations(range(len(boxes)), 2):
        if edge_dist(boxes[i], boxes[j]) <= dist_thresh:
            G.add_edge(i, j)

    merged = []
    for comp in nx.connected_components(G):
        xs = [boxes[i][0] for i in comp]
        ys = [boxes[i][1] for i in comp]
        ws = [boxes[i][2] for i in comp]
        hs = [boxes[i][3] for i in comp]
        x0, y0 = min(xs), min(ys)
        x1 = max(x + w for x, w in zip(xs, ws))
        y1 = max(y + h for y, h in zip(ys, hs))
        merged.append((x0, y0, x1 - x0, y1 - y0))
    return merged


def merge_overlapping_boxes(boxes):
    def overlaps(b1, b2):
        x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
        dx = min(x1+w1, x2+w2) - max(x1, x2)
        dy = min(y1+h1, y2+h2) - max(y1, y2)
        return (dx > 0) and (dy > 0)

    G = nx.Graph()
    G.add_nodes_from(range(len(boxes)))
    for i, j in combinations(range(len(boxes)), 2):
        if overlaps(boxes[i], boxes[j]):
            G.add_edge(i, j)

    merged = []
    for comp in nx.connected_components(G):
        xs = [boxes[i][0] for i in comp]
        ys = [boxes[i][1] for i in comp]
        ws = [boxes[i][2] for i in comp]
        hs = [boxes[i][3] for i in comp]
        x0, y0 = min(xs), min(ys)
        x1 = max(x + w for x, w in zip(xs, ws))
        y1 = max(y + h for y, h in zip(ys, hs))
        merged.append((x0, y0, x1 - x0, y1 - y0))
    return merged


def merge_row_boxes(boxes, y_thresh_factor=0.5):
    if not boxes:
        return []
    heights = [h for (_, _, _, h) in boxes]
    row_thresh = np.median(heights) * y_thresh_factor
    sorted_boxes = sorted(boxes, key=lambda b: b[1] + b[3]/2)
    clusters = []
    for b in sorted_boxes:
        yc = b[1] + b[3]/2
        if not clusters:
            clusters.append([b])
        else:
            last = clusters[-1]
            avg_yc = np.mean([bb[1] + bb[3]/2 for bb in last])
            if abs(yc - avg_yc) <= row_thresh:
                last.append(b)
            else:
                clusters.append([b])
    merged_rows = []
    for group in clusters:
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        ws = [b[2] for b in group]
        hs = [b[3] for b in group]
        x0 = min(xs)
        y0 = min(ys)
        x1 = max(x + w for x, w in zip(xs, ws))
        y1 = max(y + h for y, h in zip(ys, hs))
        merged_rows.append((x0, y0, x1 - x0, y1 - y0))
    return merged_rows


def merge_vertical_overlap_boxes(boxes):
    if not boxes:
        return []
    def vertical_overlap(b1, b2):
        y1, h1 = b1[1], b1[3]
        y2, h2 = b2[1], b2[3]
        return (min(y1+h1, y2+h2) - max(y1, y2)) > 0
    Gv = nx.Graph()
    Gv.add_nodes_from(range(len(boxes)))
    for i, j in combinations(range(len(boxes)), 2):
        if vertical_overlap(boxes[i], boxes[j]):
            Gv.add_edge(i, j)
    merged_vs = []
    for comp in nx.connected_components(Gv):
        xs = [boxes[i][0] for i in comp]
        ys = [boxes[i][1] for i in comp]
        ws = [boxes[i][2] for i in comp]
        hs = [boxes[i][3] for i in comp]
        x0, y0 = min(xs), min(ys)
        x1 = max(x + w for x, w in zip(xs, ws))
        y1 = max(y + h for y, h in zip(ys, hs))
        merged_vs.append((x0, y0, x1 - x0, y1 - y0))
    return merged_vs


def extend_to_full_width(boxes, img_width):
    return [(0, y, img_width, h) for (_, y, _, h) in boxes]

# ─── Annotation ────────────────────────────────────────────────────────────
def annotate_image(disp, boxes):
    annotated = disp.copy()
    boxes_json = []
    for idx, (x, y, w, h) in enumerate(boxes):
        border_expand = 2
        horizontal_thickness = 2
        vertical_thickness = 4
        cv2.line(annotated, (x - border_expand, y - border_expand), (x + w + border_expand, y - border_expand), (180, 0, 0), horizontal_thickness)
        cv2.line(annotated, (x - border_expand, y + h + border_expand), (x + w + border_expand, y + h + border_expand), (180, 0, 0), horizontal_thickness)
        left_x = border_expand
        right_x = annotated.shape[1] - border_expand - 1
        cv2.line(annotated, (left_x, y - border_expand), (left_x, y + h + border_expand), (180, 0, 0), vertical_thickness)
        cv2.line(annotated, (right_x, y - border_expand), (right_x, y + h + border_expand), (180, 0, 0), vertical_thickness)
        text = str(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        text_x = x + w - 160
        text_y = y + (h + text_height) // 2
        cv2.putText(annotated, text, (text_x, text_y), font, font_scale, (180, 0, 0), text_thickness)
        boxes_json.append({"id": idx, "bbox": [x, y, w, h]})
    _, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    annotated_bytes = buf.tobytes()
    return annotated, annotated_bytes, boxes_json

# ─── Main Processing ──────────────────────────────────────────────────────
def preprocess_image(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logs = ["✅ Image loaded", "✅ Converted image to RGB"]
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(disp, detail=1, paragraph=False)
    logs.append(f"✅ EasyOCR detected {len(results)} text boxes")
    boxes = []
    for bbox, text, conf in results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        boxes.append((x, y, w, h))
    logs.append(f"✅ Extracted {len(boxes)} raw bounding boxes")
    dist_thresh = np.median([h for (_, _, _, h) in boxes]) * 1.2 if boxes else 0
    boxes = merge_nearby_boxes(boxes, dist_thresh)
    logs.append(f"✅ Merged nearby boxes → {len(boxes)} boxes")
    boxes = merge_overlapping_boxes(boxes)
    logs.append(f"✅ Merged overlapping boxes → {len(boxes)} final boxes")
    boxes = merge_row_boxes(boxes, y_thresh_factor=0.5)
    logs.append(f"✅ Merged row-aligned boxes → {len(boxes)} cleaned rows")
    boxes = merge_vertical_overlap_boxes(boxes)
    logs.append(f"✅ Merged vertically overlapping boxes → {len(boxes)} cleaned verticals")
    img_width = disp.shape[1]
    boxes = extend_to_full_width(boxes, img_width)
    logs.append(f"✅ Extended all boxes to full image width → {len(boxes)} full-width boxes")
    annotated, annotated_bytes, boxes_json = annotate_image(disp, boxes)
    logs.append("✅ Annotated image with final boxes and IDs")
    return {"disp": disp, "annotated": annotated, "annotated_bytes": annotated_bytes, "boxes_json": boxes_json, "logs": logs}

def process_images(uploaded_files):
    results = []
    for file in uploaded_files:
        file_bytes = file.read()
        res = preprocess_image(file_bytes)
        res['file_name'] = file.name
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
        x_min = min(boxes_by_id[b][0] for b in box_ids if b in boxes_by_id)
        y_min = min(boxes_by_id[b][1] for b in box_ids if b in boxes_by_id)
        x_max = max(boxes_by_id[b][0] + boxes_by_id[b][2] for b in box_ids if b in boxes_by_id)
        y_max = max(boxes_by_id[b][1] + boxes_by_id[b][3] for b in box_ids if b in boxes_by_id)
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