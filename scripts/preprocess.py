#!/usr/bin/env python3
"""Phase 1: Preprocess first 500 Glassdoor records. Tasks 1, 4, 5 + industry lookup."""
import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to raw input data — set via env var or override here
SOURCE_FILE = os.environ.get("SOURCE_FILE", os.path.join(BASE_DIR, "data/source/glassdoor_raw.json"))
OUTPUT_FILE = os.path.join(BASE_DIR, "data/raw/batch_000.jsonl")
LOOKUP_FILE = os.path.join(BASE_DIR, "config/industry_lookup.json")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RECORD_LIMIT = 500

# Task 1: Subtype normalization
SUBTYPE_LABELS = {
    "glassdoorConsistentLeadershipComplaints": "Leadership Complaints",
    "glassdoorHighCulturePraise": "Culture Praise",
    "glassdoorCompensationDissatisfaction": "Compensation Dissatisfaction",
    "glassdoorGrowthOpportunities": "Growth Opportunities",
    "glassdoorPoorWorkLifeBalance": "Poor Work-Life Balance",
    "glassdoorTalentRetentionConcerns": "Talent Retention Concerns",
}

SUBTYPE_CATEGORIES = {
    "glassdoorConsistentLeadershipComplaints": "risk",
    "glassdoorHighCulturePraise": "strength",
    "glassdoorCompensationDissatisfaction": "risk",
    "glassdoorGrowthOpportunities": "risk",
    "glassdoorPoorWorkLifeBalance": "risk",
    "glassdoorTalentRetentionConcerns": "risk",
}

# Task 4: Rating labels
RATING_DISPLAY_NAMES = {
    "work_life_balance_rating": "work-life balance",
    "culture_rating": "culture",
    "management_rating": "management",
    "compensation_rating": "compensation",
    "career_opportunities_rating": "career opportunities",
}


def rate_label(score):
    if score is None:
        return None
    if score >= 4.0:
        return "strong"
    elif score >= 3.5:
        return "above average"
    elif score >= 3.0:
        return "average"
    elif score >= 2.5:
        return "below average"
    else:
        return "poor"


def rate_pct_label(score):
    if score is None:
        return None
    if score >= 0.75:
        return "high"
    elif score >= 0.55:
        return "moderate"
    elif score >= 0.35:
        return "low"
    else:
        return "very low"


def build_rating_summary(company_name, ratings):
    if not ratings:
        return None
    overall = ratings.get("overall_rating")
    if overall is None:
        return None

    strengths = []
    weaknesses = []
    for field, display in RATING_DISPLAY_NAMES.items():
        val = ratings.get(field)
        if val is None:
            continue
        label = rate_label(val)
        if label in ("strong", "above average"):
            strengths.append((display, label))
        elif label in ("below average", "poor"):
            weaknesses.append((display, label))

    ceo = ratings.get("ceo_approval")
    if ceo is not None:
        ceo_label = rate_pct_label(ceo)
        if ceo_label in ("high",):
            strengths.append(("CEO approval", ceo_label))
        elif ceo_label in ("low", "very low"):
            weaknesses.append(("CEO approval", ceo_label))

    if overall >= 3.5 and len(weaknesses) >= 2:
        sentiment = "mixed-to-positive"
    elif overall >= 3.5:
        sentiment = "generally positive"
    elif overall >= 3.0:
        sentiment = "mixed"
    elif overall >= 2.5:
        sentiment = "slightly negative"
    else:
        sentiment = "negative"

    parts = [f"Overall, employees at {company_name} are {sentiment} (rated {overall}/5)"]

    if weaknesses:
        weak_str = " and ".join(f"{label} {name}" for name, label in weaknesses[:3])
        parts.append(f"citing {weak_str}")

    if strengths and weaknesses:
        strong_str = " and ".join(f"{label} {name}" for name, label in strengths[:2])
        parts.append(f"offset in part by {strong_str}")
    elif strengths:
        strong_str = " and ".join(f"{label} {name}" for name, label in strengths[:3])
        parts.append(f"highlighting {strong_str}")

    return ", ".join(parts) + "."


# Task 5: Deterministic scores
def signal_strength(data):
    conf = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(data.get("confidence", "low"), 0.3)
    rel = data.get("relevance", 0)
    reviews = min(data.get("total_reviews", 0) / 100, 1.0)
    return round(conf * 0.4 + rel * 0.4 + reviews * 0.2, 2)


def completeness_score(record):
    score = 0
    company = record.get("company") or {}
    data = record.get("data") or {}
    if company.get("linkedin_url"):
        score += 10
    if company.get("industries"):
        score += 10
    if company.get("employee_count_low") is not None:
        score += 10
    desc = company.get("description")
    if desc and len(desc.strip()) >= 30:
        score += 10
    if data.get("job_titles"):
        score += 10
    if data.get("total_reviews", 0) >= 10:
        score += 10
    if data.get("confidence") in ("high", "medium"):
        score += 10
    if data.get("relevance", 0) >= 0.5:
        score += 10
    if data.get("competitors_mentioned"):
        score += 10
    if data.get("technologies_mentioned"):
        score += 10
    return score


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load industry lookup
    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        industry_lookup = json.load(f)
    print(f"Loaded {len(industry_lookup)} industry mappings")

    records = []
    subtype_counts = {}
    industry_hit = 0
    industry_miss = 0
    no_industry = 0
    rating_summary_count = 0

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            subtype = rec.get("signal_subtype", "")
            company = rec.get("company") or {}
            data = rec.get("data") or {}
            ratings = data.get("overallCompanyRatings") or {}

            # Task 1: Normalize subtype
            enriched = {
                "signalSubtypeNormalized": SUBTYPE_LABELS.get(subtype, subtype),
                "signalCategory": SUBTYPE_CATEGORIES.get(subtype, "unknown"),
            }

            # Task 2: Industry lookup (for records with industries)
            raw_industries = company.get("industries") or []
            industry = None
            sub_industry = None
            industry_source = None
            if raw_industries:
                label = raw_industries[0].strip() if raw_industries else ""
                if label in industry_lookup:
                    mapping = industry_lookup[label]
                    industry = mapping["industry"]
                    sub_industry = mapping.get("subIndustry")
                    industry_source = "normalized"
                    industry_hit += 1
                else:
                    industry_miss += 1
            else:
                no_industry += 1

            enriched["industry"] = industry
            enriched["subIndustry"] = sub_industry
            enriched["industrySource"] = industry_source
            enriched["industryOriginal"] = ", ".join(raw_industries) if raw_industries else None

            # Task 4: Rating summary
            rating_summary = build_rating_summary(company.get("name", "this company"), ratings)
            enriched["ratingSummary"] = rating_summary
            if rating_summary:
                rating_summary_count += 1

            # Task 5: Deterministic scores
            enriched["signalStrength"] = signal_strength(data)
            enriched["completenessScore"] = completeness_score(rec)

            rec["_enriched"] = enriched
            records.append(rec)
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

            if len(records) >= RECORD_LIMIT:
                break

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    checkpoint = {
        "total_records": len(records),
        "phase1_complete": True,
        "preprocessed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(CHECKPOINT_DIR, "progress.json"), "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"Preprocessed {len(records)} records -> {OUTPUT_FILE}")
    print(f"\nSubtype distribution:")
    for k, v in sorted(subtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(records)*100:.1f}%)")
    print(f"\nIndustry lookup: {industry_hit} hits, {industry_miss} misses, {no_industry} no-industry")
    print(f"Rating summaries generated: {rating_summary_count}/{len(records)}")


if __name__ == "__main__":
    main()
