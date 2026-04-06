#!/usr/bin/env python3
"""Phase 2: LLM worker for Glassdoor v2. Tasks 2 (missing industry) + 3 (GTM) via Qwen 3 8B."""
import json
import os
import sys
import time
import logging
from datetime import datetime, timezone
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data/raw/batch_000.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "data/output/glassdoor_v2_cleansed.jsonl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoints/progress.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")

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

ALLOWED_CATEGORIES = {
    "hr_and_people_ops", "compensation_and_benefits",
    "learning_and_development", "leadership_and_management",
    "employee_engagement", "work_life_balance", "technology_and_tools",
}

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "worker.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def call_qwen(prompt, task_name, signal_id):
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
        except json.JSONDecodeError:
            if attempt == 0:
                logger.warning(f"  JSON parse fail for {task_name} on {signal_id}, retrying")
                time.sleep(2)
            else:
                logger.error(f"  JSON parse fail for {task_name} on {signal_id} after retry: {text[:200]}")
        except requests.exceptions.RequestException as e:
            if attempt == 0:
                logger.warning(f"  Request fail for {task_name} on {signal_id}: {e}, retrying")
                time.sleep(3)
            else:
                logger.error(f"  Request fail for {task_name} on {signal_id} after retry: {e}")
    return None


def validate_industry(resp):
    if resp is None:
        return None
    if resp.get("confidence") == "low":
        return None
    if resp.get("industry") and resp["industry"] not in ALLOWED_INDUSTRIES:
        return None
    return resp


def validate_gtm(resp):
    if resp is None:
        return None
    cat = resp.get("opportunityCategory")
    if cat and cat not in ALLOWED_CATEGORIES:
        resp["opportunityCategory"] = None
        resp["gtmUseCase"] = None
    return resp


def task2_prompt(company):
    name = company.get("name", "")
    domain = company.get("domain", "")
    desc = (company.get("description") or "")[:300]
    input_data = json.dumps({"name": name, "domain": domain, "description": desc})
    return f"""Classify this company's industry based on its name, domain, and description.

RULES:
1. Return a normalized industry and sub-industry.
2. If you cannot determine the industry with reasonable confidence, return null for both.
3. NEVER guess. If the company name is generic and there is no description, return null.
4. Base your classification ONLY on the information provided. Do not use outside knowledge about the company.

ALLOWED INDUSTRIES: technology, finance, healthcare, manufacturing, retail, education, media_and_entertainment, real_estate, energy, transportation, government, professional_services, food_and_beverage, telecommunications, agriculture, construction, nonprofit, hospitality, automotive, aerospace_and_defense

INPUT: {input_data}

Respond with ONLY valid JSON. No markdown, no explanation.
FORMAT: {{"industry": "parent_category or null", "subIndustry": "specific_sub or null", "confidence": "high|medium|low"}}"""


def task3_prompt(rec):
    data = rec.get("data") or {}
    company = rec.get("company") or {}
    input_data = json.dumps({
        "signal_subtype": rec.get("signal_subtype", ""),
        "summary": (data.get("summary") or "")[:500],
        "detail": (data.get("detail") or "")[:800],
        "industries": company.get("industries") or [],
        "description": (company.get("description") or "")[:200],
    })
    return f"""You are a B2B sales analyst. Given a Glassdoor employee review signal, determine if it reveals a problem that a third-party vendor could solve.

OPPORTUNITY CATEGORIES (pick the best match, or null):
- "hr_and_people_ops": No HR department, poor hiring processes, lack of structure
- "compensation_and_benefits": Below-market pay, poor benefits, no raises
- "learning_and_development": No training, limited career growth, no mentorship
- "leadership_and_management": Poor leadership, no direction, micromanagement
- "employee_engagement": Low morale, high turnover, poor retention
- "work_life_balance": Burnout, overwork, no flexibility
- "technology_and_tools": Outdated tools, lack of software, manual processes
- null: No clear selling opportunity (e.g., positive culture praise, vague feedback)

RULES:
1. The opportunity MUST be grounded in the summary and detail text. Do not infer problems not mentioned.
2. Write a specific, 1-sentence use case from the perspective of a salesperson. Example: "Company lacks formal HR processes - opportunity to sell HR management software."
3. If the signal is positive (culture praise, good management), return null UNLESS it mentions a specific gap.
4. If the detail is too vague to identify a specific need, return null. Err on the side of null.

INPUT: {input_data}

Respond with ONLY valid JSON. No markdown, no explanation.
FORMAT: {{"gtmUseCase": "specific 1-sentence opportunity or null", "opportunityCategory": "category from list above or null"}}"""


def process_record(rec, index):
    signal_id = rec.get("signal_id", f"unknown_{index}")
    company = rec.get("company") or {}
    data = rec.get("data") or {}
    enriched = rec.get("_enriched", {})
    ratings = data.get("overallCompanyRatings") or {}
    errors = []
    tasks_run = []

    # Industry from preprocessing (lookup)
    industry = enriched.get("industry")
    sub_industry = enriched.get("subIndustry")
    industry_source = enriched.get("industrySource")
    industry_original = enriched.get("industryOriginal")

    # Task 2: If no industry from lookup, classify via Qwen
    if not industry:
        t2_raw = call_qwen(task2_prompt(company), "industry", signal_id)
        t2 = validate_industry(t2_raw)
        tasks_run.append("industry")
        if t2 and t2.get("industry"):
            industry = t2["industry"]
            sub_industry = t2.get("subIndustry")
            industry_source = "classified"
        if t2_raw is None:
            errors.append("task2_parse_fail")

    # Task 3: GTM use case (always)
    t3_raw = call_qwen(task3_prompt(rec), "gtmUseCase", signal_id)
    t3 = validate_gtm(t3_raw)
    tasks_run.append("gtmUseCase")

    gtm_use_case = None
    opp_category = None
    if t3:
        gtm_use_case = t3.get("gtmUseCase")
        opp_category = t3.get("opportunityCategory")
        # Null string handling
        if gtm_use_case in ("null", "None", ""):
            gtm_use_case = None
        if opp_category in ("null", "None", ""):
            opp_category = None
    if t3_raw is None:
        errors.append("task3_parse_fail")

    return {
        "signalId": signal_id,
        "signalType": rec.get("signal_type"),
        "signalSubtype": rec.get("signal_subtype"),
        "signalSubtypeNormalized": enriched.get("signalSubtypeNormalized"),
        "signalCategory": enriched.get("signalCategory"),
        "detectedAt": rec.get("detected_at"),
        "companyName": company.get("name"),
        "companyDomain": company.get("domain"),
        "companyLinkedinUrl": company.get("linkedin_url"),
        "industry": industry,
        "subIndustry": sub_industry,
        "industryOriginal": industry_original,
        "industrySource": industry_source,
        "summary": data.get("summary"),
        "detail": data.get("detail"),
        "relevance": data.get("relevance"),
        "confidence": data.get("confidence"),
        "sentiment": data.get("sentiment"),
        "overallRating": ratings.get("overall_rating"),
        "workLifeBalanceRating": ratings.get("work_life_balance_rating"),
        "cultureRating": ratings.get("culture_rating"),
        "managementRating": ratings.get("management_rating"),
        "compensationRating": ratings.get("compensation_rating"),
        "careerOpportunitiesRating": ratings.get("career_opportunities_rating"),
        "ceoApproval": ratings.get("ceo_approval"),
        "recommendToFriend": ratings.get("recommend_to_friend"),
        "businessOutlook": ratings.get("business_outlook"),
        "ratingSummary": enriched.get("ratingSummary"),
        "gtmUseCase": gtm_use_case,
        "opportunityCategory": opp_category,
        "totalReviews": data.get("total_reviews"),
        "recentReviewsCount": data.get("recent_reviews_count"),
        "jobTitles": data.get("job_titles") or [],
        "glassdoorId": data.get("glassdoor_id"),
        "glassdoorUrl": data.get("glassdoor_url"),
        "competitorsMentioned": data.get("competitors_mentioned") or [],
        "technologiesMentioned": data.get("technologies_mentioned") or [],
        "_meta": {
            "completenessScore": enriched.get("completenessScore", 0),
            "signalStrength": enriched.get("signalStrength", 0),
            "processedAt": datetime.now(timezone.utc).isoformat(),
            "qwenModel": MODEL,
            "tasksRun": tasks_run,
            "errors": errors,
        },
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} records from {INPUT_FILE}")
    no_industry = sum(1 for r in records if not r.get("_enriched", {}).get("industry"))
    logger.info(f"Records needing industry classification: {no_industry}")
    logger.info(f"All records get GTM use case call")

    start_time = time.time()
    processed = []

    for i, rec in enumerate(records):
        result = process_record(rec, i)
        processed.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 3600
            eta_min = (len(records) - i - 1) / (rate / 60) if rate > 0 else 0
            logger.info(f"  {i+1}/{len(records)} done ({rate:.0f} rec/hr, ~{eta_min:.0f} min remaining)")
            sys.stdout.flush()

            checkpoint = {
                "last_processed": i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(checkpoint, f, indent=2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    logger.info(f"Done. {len(processed)} records in {elapsed/60:.1f} min. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
