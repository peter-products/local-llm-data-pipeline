#!/usr/bin/env python3
"""Build industry lookup table from unique Glassdoor industry labels via Qwen 3 8B."""
import json
import os
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.environ.get("SOURCE_FILE", os.path.join(BASE_DIR, "data/source/glassdoor_raw.json"))
OUTPUT_FILE = os.path.join(BASE_DIR, "config/industry_lookup.json")
RECORD_LIMIT = 500

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

ALLOWED_INDUSTRIES = {
    "technology", "finance", "healthcare", "manufacturing", "retail",
    "education", "media_and_entertainment", "real_estate", "energy",
    "transportation", "government", "professional_services",
    "food_and_beverage", "telecommunications", "agriculture",
    "construction", "nonprofit", "hospitality", "automotive",
    "aerospace_and_defense",
}


def call_qwen(prompt):
    for attempt in range(2):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            }, timeout=30)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()

            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            return json.loads(text)
        except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
            if attempt == 0:
                time.sleep(2)
            else:
                print(f"  Failed after retry: {e}")
    return None


def build_prompt(label):
    return f"""You normalize industry labels. Convert the given Glassdoor industry label to a standardized format.

RULES:
1. Map to the closest matching parent industry from the allowed list.
2. Create a specific sub-industry in lowercase_snake_case.
3. If the label is ambiguous or could map to multiple industries, pick the most common interpretation.

ALLOWED INDUSTRIES: technology, finance, healthcare, manufacturing, retail, education, media_and_entertainment, real_estate, energy, transportation, government, professional_services, food_and_beverage, telecommunications, agriculture, construction, nonprofit, hospitality, automotive, aerospace_and_defense

EXAMPLES:
- "It Services And It Consulting" -> {{"industry": "technology", "subIndustry": "it_services_and_consulting"}}
- "Banking & Lending" -> {{"industry": "finance", "subIndustry": "banking_and_lending"}}
- "Wellness And Fitness Services" -> {{"industry": "healthcare", "subIndustry": "wellness_and_fitness"}}
- "Building Construction" -> {{"industry": "construction", "subIndustry": "building_construction"}}
- "Non-Profit Organizations" -> {{"industry": "nonprofit", "subIndustry": "general"}}
- "Museums, Historical Sites, And Zoos" -> {{"industry": "media_and_entertainment", "subIndustry": "museums_and_cultural_institutions"}}

INPUT: {{"glassdoor_industry": "{label}"}}

Respond with ONLY valid JSON. No markdown, no explanation.
FORMAT: {{"industry": "parent_category", "subIndustry": "specific_sub", "confidence": "high|medium|low"}}"""


def main():
    # Collect unique industry labels from first 500 records
    unique_labels = set()
    count = 0
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            industries = rec.get("company", {}).get("industries") or []
            for label in industries:
                if label and label.strip():
                    unique_labels.add(label.strip())
            count += 1
            if count >= RECORD_LIMIT:
                break

    print(f"Scanned {count} records, found {len(unique_labels)} unique industry labels")

    # Process each through Qwen
    lookup = {}
    failed = []
    for i, label in enumerate(sorted(unique_labels)):
        resp = call_qwen(build_prompt(label))
        if resp and resp.get("confidence") != "low" and resp.get("industry") in ALLOWED_INDUSTRIES:
            lookup[label] = {
                "industry": resp["industry"],
                "subIndustry": resp.get("subIndustry"),
            }
            print(f"  [{i+1}/{len(unique_labels)}] {label} -> {resp['industry']} / {resp.get('subIndustry')}")
        else:
            failed.append(label)
            print(f"  [{i+1}/{len(unique_labels)}] {label} -> FAILED ({resp})")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(lookup)} mappings to {OUTPUT_FILE}")
    if failed:
        print(f"Failed labels ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
