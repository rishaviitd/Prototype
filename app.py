import os
import warnings
import logging
import sys
from PIL import Image, ImageOps
import io
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

# ─── Suppress noisy errors/warnings ──────────────────────────────────────────
# Disable Streamlit's file-watcher to avoid Torch "__path__._path" errors
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_WATCH_PATHS"] = "false"  # Additional watcher config

# Fix asyncio RuntimeError: no running event loop
if sys.platform == 'darwin':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Ignore the pin_memory warning from PyTorch DataLoader on MPS
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
# Ignore torch.__path__._path errors
warnings.filterwarnings("ignore", message=".*__path__._path.*", category=RuntimeWarning)

# Silence Streamlit watcher / bootstrap logs about asyncio & torch.classes
logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.web.bootstrap").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

# ─── Imports ────────────────────────────────────────────────────────────────
import streamlit as st
import cv2
import numpy as np
import easyocr
import networkx as nx
import json
from itertools import combinations
from math import sqrt
from agent import call_gemini_api, call_gemini_match_api  # Import the Gemini API functions
import orchestration

# ─── Function to crop questions based on boxes ───────────────────────────
def crop_questions(image, boxes_json, question_mapping):
    """
    Crop sections of the image corresponding to each question based on box mapping.
    
    Args:
        image: The original image (RGB)
        boxes_json: List of dicts with box info (id and bbox)
        question_mapping: Dict mapping question IDs to box IDs
        
    Returns:
        Dict mapping question IDs to cropped images
    """
    # Convert box IDs to integers in question mapping
    for q_id, box_ids in question_mapping.items():
        question_mapping[q_id] = [int(box_id) for box_id in box_ids]
    
    # Create a lookup for boxes by ID
    boxes_by_id = {box['id']: box['bbox'] for box in boxes_json}
    
    # Compute original bounding rectangles for each question
    height, width = image.shape[:2]
    regions = []
    for q_id, box_ids in question_mapping.items():
        if not box_ids:
            continue
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        for box_id in box_ids:
            if box_id not in boxes_by_id:
                continue
            x, y, w, h = boxes_by_id[box_id]
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        regions.append({'q_id': q_id, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})

    # Sort regions by vertical position
    regions.sort(key=lambda r: r['y_min'])

    # Adjust boundaries and crop each region
    cropped_images = {}
    for idx, region in enumerate(regions):
        y_min, y_max = region['y_min'], region['y_max']
        # Top boundary
        if idx > 0:
            prev_y_max = regions[idx-1]['y_max']
            half_gap = (y_min - prev_y_max) / 2.0
            y_min_adj = int(max(0, y_min - half_gap))
        else:
            y_min_adj = int(y_min)
        # Bottom boundary
        if idx < len(regions) - 1:
            next_y_min = regions[idx+1]['y_min']
            half_gap = (next_y_min - y_max) / 2.0
            y_max_adj = int(min(height, y_max + half_gap))
        else:
            y_max_adj = int(y_max)
        x1, x2 = region['x_min'], region['x_max']
        if x1 < x2 and y_min_adj < y_max_adj:
            cropped_images[region['q_id']] = image[y_min_adj:y_max_adj, x1:x2]

    return cropped_images

# ─── Streamlit page config ─────────────────────────────────────────────────
st.set_page_config(page_title="OCR & Box Merge App")

st.title("OCR & Box Merge Streamlit App")

# Orchestration for multi-image upload and processing
uploaded_files = st.file_uploader(
    "Upload PNG or JPEG images", type=["png","jpg","jpeg"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one image to begin.")
    st.stop()

with st.spinner("Running OCR on uploaded images..."):
    processed_results = orchestration.process_images(uploaded_files)
    # Store processed results for later use in puzzle solving
    st.session_state['processed_results'] = processed_results

# Initialize session state containers
if 'gemini_responses' not in st.session_state:
    st.session_state['gemini_responses'] = {}
if 'cropped_questions' not in st.session_state:
    st.session_state['cropped_questions'] = {}

# Create tabs, including Final for puzzle results
tab_logs, tab_result, tab_gemini, tab_crops, tab_final = st.tabs([
    "Logs", "Result", "Gemini Analysis", "Cropped Questions", "Final"
])

with tab_logs:
    for res in processed_results:
        st.subheader(res['file_name'])
        st.text("\n".join(res['logs']))

with tab_result:
    for res in processed_results:
        st.subheader(res['file_name'])
        st.image(res['annotated'], caption="Annotated Image", use_container_width=True)
        st.download_button(
            "⬇️ Download Annotated Image",
            res['annotated_bytes'],
            file_name=f"{res['file_name']}_annotated.png",
            mime="image/png",
            key=f"download_img_{res['file_name']}"
        )
        st.subheader("Detected Boxes JSON")
        st.json(res['boxes_json'])
        json_str = json.dumps(res['boxes_json'], indent=2)
        st.download_button(
            "⬇️ Download Boxes JSON",
            json_str,
            file_name=f"{res['file_name']}_boxes.json",
            mime="application/json",
            key=f"download_json_{res['file_name']}"
        )
    if st.button("🔍 Analyze with Gemini", key="analyze_all"):
        with st.spinner("Analyzing images with Gemini AI..."):
            api_key = os.getenv("GOOGLE_API_KEY")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_name = {
                    executor.submit(call_gemini_api, res['annotated_bytes'], api_key): res['file_name']
                    for res in processed_results
                }
                for future in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[future]
                    st.session_state['gemini_responses'][name] = future.result()
            st.success("Analysis complete! Check the Gemini Analysis tab.")

with tab_gemini:
    for file_name, gemini_data in st.session_state['gemini_responses'].items():
        st.subheader(file_name)
        if "error" in gemini_data:
            st.error(f"Error: {gemini_data['error']}")
        else:
            st.json(gemini_data)
    if st.button("✂️ Crop All Questions", key="crop_all"):
        with st.spinner("Cropping question images..."):
            for res in processed_results:
                mapping = st.session_state['gemini_responses'].get(res['file_name'], {})
                cropped = orchestration.crop_questions(
                    res['disp'], res['boxes_json'], mapping
                )
                st.session_state['cropped_questions'][res['file_name']] = cropped
            st.success("Questions cropped! Check the Cropped Questions tab.")

with tab_crops:
    for file_name, cropped_images in st.session_state['cropped_questions'].items():
        st.subheader(file_name)
        if not cropped_images:
            st.warning("No cropped images available.")
        else:
            for q_id, img_crop in sorted(cropped_images.items()):
                st.subheader(f"Question {q_id}")
                st.image(img_crop, use_column_width=True)
                _, buf = cv2.imencode(
                    ".png", cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                )
                bytes_ = buf.tobytes()
                st.download_button(
                    f"⬇️ Download {file_name}_Q{q_id}",
                    bytes_,
                    file_name=f"{file_name}_question_{q_id}.png",
                    mime="image/png",
                    key=f"download_crop_{file_name}_{q_id}"
                )
    if st.button("🧩 Puzzle Solve", key="puzzle_solve"):
        st.markdown("#### 🧩 Puzzle Solve Started")
        puzzle_results = {}
        with st.spinner("Running snippet matching..."):
            api_key = os.getenv("GOOGLE_API_KEY")
            # Gather all defined snippet crops across files
            defined_snippets = []
            for src_file, crops in st.session_state['cropped_questions'].items():
                for lbl, img in crops.items():
                    if lbl != 'undefined':
                        defined_snippets.append((src_file, lbl, img))
            # Iterate over each file's undefined snippet
            for file_name, crops in st.session_state['cropped_questions'].items():
                st.info(f"▶️ Starting puzzle solve for **{file_name}**")
                has_undef = 'undefined' in crops
                st.info(f"   🔢 Defined snippet images: {len(defined_snippets)}")
                st.info(f"   🔢 Undefined snippet images: {1 if has_undef else 0}")
                if not has_undef:
                    st.warning(f"⏹️ No undefined snippets found in **{file_name}**, skipping.")
                    continue
                undef_img = crops['undefined']
                matched_page = None
                # Match against all defined snippets until hit
                for src_file_def, q_label, def_img in list(defined_snippets):
                    st.info(f"🔍 Checking undefined snippet of **{file_name}** against question **{q_label}** from **{src_file_def}**")
                    # Encode images to PNG bytes
                    _, buf_u = cv2.imencode('.png', cv2.cvtColor(undef_img, cv2.COLOR_RGB2BGR))
                    _, buf_d = cv2.imencode('.png', cv2.cvtColor(def_img, cv2.COLOR_RGB2BGR))
                    match_res = call_gemini_match_api(buf_u.tobytes(), buf_d.tobytes(), api_key)
                    result = match_res.get('match', 'error')
                    st.write(f"➡️ Result for **{q_label}**: **{result.upper()}**")
                    if result == 'yes':
                        st.success(f"✅ Match confirmed: undefined snippet of **{file_name}** matches **{q_label}** from **{src_file_def}**. Composing combined image...")
                        # Stack images vertically
                        h_u, w_u = undef_img.shape[:2]
                        h_d, w_d = def_img.shape[:2]
                        new_h, new_w = h_d + h_u, max(w_d, w_u)
                        canvas = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
                        x_off_d = (new_w - w_d) // 2
                        x_off_u = (new_w - w_u) // 2
                        canvas[:h_d, x_off_d:x_off_d+w_d] = def_img
                        canvas[h_d:h_d+h_u, x_off_u:x_off_u+w_u] = undef_img
                        puzzle_results.setdefault(file_name, []).append(canvas)
                        matched_page = src_file_def
                        break
                if matched_page is not None:
                    # Remove all defined snippets from that page
                    defined_snippets = [snip for snip in defined_snippets if snip[0] != matched_page]
        st.session_state['puzzle_results'] = puzzle_results
        st.success("🎉 Puzzle solving complete! Navigate to the Final tab to see results.")

# Final tab to display puzzle results
with tab_final:
    st.header("Puzzle Solve Final Results")
    puzzle_results = st.session_state.get('puzzle_results', {})
    if puzzle_results:
        for file_name, images in puzzle_results.items():
            st.subheader(file_name)
            for img in images:
                st.image(img, use_container_width=True)
    else:
        st.info("No puzzle results available. Please run Puzzle Solve.")
