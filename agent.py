import requests
import json
import base64
from typing import Dict, List, Any

def call_gemini_api(image_bytes: bytes, api_key: str) -> Dict[str, List[int]]:
    """
    Call the Gemini 2.0 Flash API with the annotated image and return the structured output.
    
    Args:
        image_bytes: The image bytes to send to the API
        api_key: Gemini API key
        
    Returns:
        Dict mapping question numbers to box IDs
    """
    # Base64 encode the image
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create the request payload
    prompt = """
    Here is a scanned page of a student's answer sheet. The page is annotated with non-overlapping red bounding boxes, each marked with a unique integer box ID in red. Boxes may:

    - Contain explicit question labels (e.g., "Q1", "Q2"),
    - Be part of a continued answer without a label,
    - Be **completely blank** (e.g., due to OCR capturing header/footer/margin areas with no handwriting)

    ---

    **Phase 1: Detection**

    1. **Box IDs**  
       - Enumerate **every** visible bounding box ID in **top-to-bottom order**.  
       - Do **not** skip or invent any ID.

    2. **Question labels**  
       - Identify all distinct question indicators found inside the boxes. A box is considered labeled if it contains any of the following or similar (case-insensitive, with or without spacing or punctuation):
         - "Qn", "Q n", "Question n", "ques n", "ans n", "answer n", "soln n", "solution n", "a n"  
         where **n** is any positive integer.
       - Convert all such indicators to the standard label format `"Qn"` for consistency.
       - Do **not** invent question numbers that do not appear.

    3. **Blank boxes (including margin boxes)**  
       - A box is **blank** if it contains **no student handwriting** and **no question label**.  
       - This includes boxes that are "sitting in the margin" (e.g., top header or outer edges), unless they contain handwriting.

    ---

    **Phase 2: Assign**  
    Assign **every box ID** to **exactly one** label—either a visible question label (`"Qn"`) or `"undefined"`—following these rules:

    1. **Explicit label detection**  
       - If a box contains one of the question indicators listed above, assign it to `"Qn"`.

    2. **Continuation boxes**  
       - If a box does not contain a label but appears to continue the answer flow of its nearest **preceding** labeled box, assign it to that same `"Qn"`.

    3. **Top-of-page or disconnected content**  
       - If one or more unlabeled boxes appear **before any labeled box** or cannot clearly be linked to a preceding question, assign them to `"undefined"`.

    4. **Blank boxes (including margin boxes)**  
       - **Always** assign a blank box to the label of the **next** non-blank box **below** it.  
       - If it is the **last** box on the page, assign it to the label of the **preceding** box.  
       - This ensures margin/header/footer boxes are correctly included in context without being dropped.

    5. **No invented labels**  
       - Only use `"Qn"` labels that are present or supported by a visible question indicator. Do **not** guess or fabricate question numbers.

    6. **Full coverage requirement**  
       - Every box ID from Phase 1 must appear **exactly once** in the final JSON output. No omissions are allowed.

    **Final Output:**  
    Return **only** a valid JSON object. Each key should be a detected question label (e.g., `"Q1"`, `"Q2"`) or `"undefined"`. Each value must be an array of box IDs assigned to that label, listed in top-to-bottom order.

    Example format:
    ```json
    {
      "Q1": [0, 1, 2],
      "Q2": [3, 4],
      "undefined": [5, 6]
    }
    ```
    """
    
    # API endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
    
    # Prepare request headers and parameters
    headers = {
        "Content-Type": "application/json"
    }
    
    params = {
        "key": api_key
    }
    
    # Prepare request body
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.1,
            "top_p": 0.95,
            "response_mime_type": "application/json"
        }
    }
    
    # Make the API call
    response = requests.post(url, headers=headers, params=params, json=data)
    
    # Check if the response was successful
    if response.status_code != 200:
        return {"error": f"API Error: {response.status_code} - {response.text}"}
    
    # Parse the response
    try:
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                for part in result["candidates"][0]["content"]["parts"]:
                    if "text" in part:
                        # Extract JSON from the text
                        text = part["text"]
                        # Find JSON content between ```json and ```
                        json_start = text.find('{')
                        json_end = text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = text[json_start:json_end]
                            return json.loads(json_str)
        
        # If we couldn't parse the expected JSON format
        return {"error": "Failed to parse JSON from Gemini response"}
    except Exception as e:
        return {"error": f"Error processing response: {str(e)}"}

def call_gemini_match_api(undef_bytes: bytes, defined_bytes: bytes, api_key: str) -> Any:
    """
    Call the Gemini 2.5 Flash API to determine if an undefined snippet matches a defined snippet.
    Returns a dict like {"match": "yes"} or {"match": "no"}.
    """
    # Base64 encode both images
    base64_undef = base64.b64encode(undef_bytes).decode('utf-8')
    base64_def = base64.b64encode(defined_bytes).decode('utf-8')
    # Prompt for matching
    prompt = (
        "I will provide two image snippets: the first is an UNDEFINED answer snippet from a student's exam, "
        "and the second is a DEFINED answer snippet labeled for a question. "
        "Do these two snippets belong to the SAME question? "
        "Respond ONLY with a JSON object like {\"match\": \"yes\"} or {\"match\": \"no\"}."
    )
    # API endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64_undef}},
                    {"inline_data": {"mime_type": "image/png", "data": base64_def}}
                ]
            }
        ],
        "generation_config": {"temperature": 0.1, "top_p": 0.95, "response_mime_type": "application/json"}
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    if response.status_code != 200:
        return {"error": f"API Error: {response.status_code} - {response.text}"}
    try:
        result = response.json()
        # Extract text from candidates
        if "candidates" in result and result["candidates"]:
            parts = result["candidates"][0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
            # Parse JSON from text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        return {"error": "Failed to parse JSON from Gemini response"}
    except Exception as e:
        return {"error": f"Error processing response: {str(e)}"} 