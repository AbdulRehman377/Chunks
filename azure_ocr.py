import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
API_VERSION = "2024-07-31-preview"  # best latest
MODEL_ID = "prebuilt-layout"

PDF_PATH = "TAX_INVOICE.pdf"
OUTPUT_JSON = "RAW_OCR.json"   # where chunks will be saved


def analyze_layout_rest(file_path: str) -> dict:
    """Send file to Azure Layout model & return RAW JSON response."""
    if not ENDPOINT or not KEY:
        raise RuntimeError("Missing Azure credentials")

    analyze_url = (
        f"{ENDPOINT}/documentintelligence/documentModels/"
        f"{MODEL_ID}:analyze?api-version={API_VERSION}&outputContentFormat=markdown"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Content-Type": "application/pdf",
    }

    print("\n📡 Sending to Azure Layout Model…")
    with open(file_path, "rb") as f:
        resp = requests.post(analyze_url, headers=headers, data=f)

    if resp.status_code != 202:
        print(resp.text)
        resp.raise_for_status()

    operation_url = resp.headers.get("operation-location")
    print("⏳ Processing…")

    while True:
        poll = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": KEY})
        poll.raise_for_status()
        result = poll.json()
        
        status = result.get("status")
        if status == "succeeded":
            print("✅ Layout extraction complete!")
            return result
        elif status == "failed":
            raise RuntimeError("❌ Layout model failed")

        time.sleep(2)


if __name__ == "__main__":
    result = analyze_layout_rest(PDF_PATH)


    # Save for manual review
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("\n====================== MARKDOWN CONTENT ======================\n")
    print(result.get("analyzeResult", {}).get("content", "No content found"))
