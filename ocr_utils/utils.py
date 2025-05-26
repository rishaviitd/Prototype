import cv2
import numpy as np
import networkx as nx
from itertools import combinations
from math import sqrt

def merge_nearby_boxes(boxes, dist_thresh):
    def edge_dist(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
        dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
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
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        dx = min(x1 + w1, x2 + w2) - max(x1, x2)
        dy = min(y1 + h1, y2 + h2) - max(y1, y2)
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
        return (min(y1 + h1, y2 + h2) - max(y1, y2)) > 0
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