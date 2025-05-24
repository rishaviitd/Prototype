# OCR & Box Merge Streamlit App

A Streamlit application for performing OCR on scanned pages, merging bounding boxes, and leveraging Gemini AI for analysis.

## Prerequisites

- Python 3.8 or higher
- Git

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The application uses the Google Gemini API. By default, an API key is hard-coded in `agent.py`. To use your own key, update the `api_key` variable in `agent.py`, or set an environment variable and modify the code to read it:

```bash
export GOOGLE_API_KEY=YOUR_API_KEY
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Upload PNG or JPEG images through the web interface and follow the on-screen prompts to analyze and crop question images.

## Deactivation

When finished, you can deactivate the virtual environment:

```bash
deactivate
```
