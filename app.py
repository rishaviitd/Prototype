import os
import warnings
import logging
import sys
from PIL import Image, ImageOps, ImageDraw
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
import json
from agent import call_gemini_api, call_gemini_match_api, call_gemini_extract_text  # Import the Gemini API functions
from approach1 import orchestration as orch1
from approach2 import orchestration as orch2

# ─── Streamlit page config ─────────────────────────────────────────────────
st.set_page_config(page_title="OCR & Box Merge App")

st.title("Tick AI")

# ─── Playground Mode Toggle ─────────────────────────────────────────────────
if 'playground_mode' not in st.session_state:
    st.session_state['playground_mode'] = False
if st.button("🛠 Playground", key="play_toggle"):
    st.session_state['playground_mode'] = True
if st.session_state['playground_mode']:
    import io as _io  # for Playground only
    from PIL import Image as _Image
    # Initialize feature states
    if 'cap_tap_active' not in st.session_state:
        st.session_state['cap_tap_active'] = False
    st.header("🔬 Playground Interface")
    st.write("Welcome to the Playground! Here you can run weird experiments.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cap-Tap", key="play_cap_tap"):
            st.session_state['cap_tap_active'] = True
    with col2:
        pass

    # Cap-Tap feature: stack two images vertically
    if st.session_state['cap_tap_active']:
        st.subheader("Cap-Tap: Stack Two Images")
        files = st.file_uploader("Upload exactly 2 images to stack", type=["png","jpg","jpeg"], accept_multiple_files=True)
        if files and len(files) == 2:
            imgs = []
            for f in files:
                content = f.read()
                img = _Image.open(_io.BytesIO(content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                imgs.append(img)
            # Create canvas for stacked image
            width = max(im.width for im in imgs)
            total_height = sum(im.height for im in imgs)
            canvas = _Image.new('RGB', (width, total_height), 'white')
            y_offset = 0
            for im in imgs:
                x = (width - im.width) // 2
                canvas.paste(im, (x, y_offset))
                y_offset += im.height
            # Display and download
            buf = _io.BytesIO()
            canvas.save(buf, format='PNG')
            byte_img = buf.getvalue()
            st.image(canvas, caption="Stacked Image", use_column_width=True)
            st.download_button("Download Stacked Image", byte_img, file_name="stacked.png", mime="image/png")
        st.stop()

if 'gemini_cost' not in st.session_state:
    st.session_state['gemini_cost'] = 0.0
if 'gemini_usage' not in st.session_state:
    st.session_state['gemini_usage'] = {}

col1, col2 = st.columns([3, 1])
# Placeholder to display and update INR cost dynamically
cost_placeholder = col2.empty()
# Display initial INR cost directly (gemini_cost is in INR)
rupee_cost = st.session_state['gemini_cost']
cost_placeholder.metric("Gemini Cost (INR)", f"₹{rupee_cost:.13f}")

# Approach selection
approach = st.selectbox("Select Approach", ["Approach-1", "Approach-2"], key="approach_select")
if approach == "Approach-1":
    orchestration = orch1
else:
    orchestration = orch2

# Orchestration for multi-image upload and processing
uploaded_files = st.file_uploader(
    "Upload PNG or JPEG images", type=["png","jpg","jpeg"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one image to begin.")
    st.stop()

# Unified processing based on selected approach
if approach == "Approach-1":
    spinner_text = "Running Document AI on uploaded images..."
    with st.spinner(spinner_text):
        processed_results = orchestration.process_images(uploaded_files)
        st.session_state['processed_results'] = processed_results
elif approach == "Approach-2":
    # Perform margin crop stage and display results
    margin_crops = orchestration.margin_crop_images(uploaded_files)
    st.subheader("📐 Margin Crop Results")
    for file_name, (crop_img, margin_x, margin_sum) in margin_crops.items():
        st.subheader(file_name)
        disp_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        st.image(disp_img, caption=f"Cropped at x={margin_x}px", use_column_width=True)
        st.write(f"Detected margin at x = {margin_x}px (sum={margin_sum})")
    if st.button("🗜️ Send Cropped Images to Document AI", key="process_doc_ai"):
        with st.spinner("Running Document AI on cropped images..."):
            processed_results = orchestration.process_images(uploaded_files)
            st.session_state['processed_results'] = processed_results
    processed_results = st.session_state.get('processed_results', [])
else:
    processed_results = []

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
        # Save annotated image to local downloads directory
        output_dir = os.path.join(os.getcwd(), "downloads", "annotated-test-dataset")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{res['file_name']}_annotated.png")
        with open(save_path, "wb") as f:
            f.write(res['annotated_bytes'])
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
                    response = future.result()
                    # Handle errors
                    if isinstance(response, dict) and 'error' in response:
                        st.session_state['gemini_responses'][name] = {'error': response['error']}
                    else:
                        # Unpack mapping and usage metadata
                        mapping = response.get('mapping', {})
                        usage = response.get('usage_metadata', {})
                        st.session_state['gemini_responses'][name] = mapping
                        st.session_state['gemini_usage'][name] = usage
                        # Calculate cost in INR: $0.15 per 1M input tokens + $3.5 per 1M output tokens
                        input_tokens = usage.get('promptTokenCount', 0)+280
                        output_tokens = usage.get('candidatesTokenCount', usage.get('completionTokens', 0))
                        usd_cost = input_tokens * (0.15 / 1_000_000) + output_tokens * (3.5 / 1_000_000)
                        cost_inr = usd_cost * 85
                        st.session_state['gemini_cost'] += cost_inr
                        # Update displayed INR cost after calculation
                        cost_placeholder.metric("Gemini Cost (INR)", f"₹{st.session_state['gemini_cost']:.13f}")
            st.success("Analysis complete! Check the Gemini Analysis tab.")

with tab_gemini:
    for file_name, gemini_data in st.session_state['gemini_responses'].items():
        st.subheader(file_name)
        if "error" in gemini_data:
            st.error(f"Error: {gemini_data['error']}")
        else:
            # Display token usage for each image
            usage = st.session_state['gemini_usage'].get(file_name, {})
            # Determine input and output tokens from usage metadata
            input_tokens = usage.get('promptTokenCount', 0)+280
            output_tokens = usage.get('candidatesTokenCount', usage.get('completionTokens', 0))
            col_a, col_b = st.columns(2)
            col_a.metric("Input Tokens", input_tokens)
            col_b.metric("Output Tokens", output_tokens)
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
                    # Calculate and update cost for this match API call
                    usage_match = match_res.get('usage_metadata', {})
                    # Approximate input and output tokens and compute cost
                    input_tokens_m = usage_match.get('promptTokenCount', 0) + 280
                    output_tokens_m = usage_match.get('candidatesTokenCount', usage_match.get('completionTokens', 0))
                    usd_cost_m = input_tokens_m * (0.15 / 1_000_000) + output_tokens_m * (3.5 / 1_000_000)
                    cost_inr_m = usd_cost_m * 85
                    st.session_state['gemini_cost'] += cost_inr_m
                    cost_placeholder.metric("Gemini Cost (INR)", f"₹{st.session_state['gemini_cost']:.13f}")
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
