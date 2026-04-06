#!/usr/bin/env python3
"""Phase 3: Validation script for Glassdoor v2. Reports on all 5 tasks."""
import json
import os
from collections import Counter
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data/output/glassdoor_v2_cleansed.jsonl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def load_records():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def check_range(name, value, low, high):
    if value < low:
        return f"BELOW RANGE ({value:.1%} < {low:.0%})"
    elif value > high:
        return f"ABOVE RANGE ({value:.1%} > {high:.0%})"
    return f"OK ({value:.1%})"


def validate(records):
    total = len(records)
    if total == 0:
        return "No records found.", []

    flags = []

    def check(name, value, low, high):
        status = check_range(name, value, low, high)
        if "RANGE" in status:
            flags.append(f"{name}: {status}")
        return status

    # --- Task 1: Subtype normalization ---
    subtypes = Counter(r.get("signalSubtypeNormalized") for r in records)
    categories = Counter(r.get("signalCategory") for r in records)

    # --- Task 2: Industry ---
    industry_assigned = sum(1 for r in records if r.get("industry"))
    industry_rate = industry_assigned / total
    industry_sources = Counter(r.get("industrySource") for r in records if r.get("industrySource"))
    ind_dist = Counter(r.get("industry") for r in records if r.get("industry"))

    # --- Task 3: GTM Use Case ---
    gtm_nonnull = sum(1 for r in records if r.get("gtmUseCase"))
    gtm_rate = gtm_nonnull / total
    opp_cats = Counter(r.get("opportunityCategory") for r in records if r.get("opportunityCategory"))

    # GTM by subtype
    subtype_gtm = {}
    for r in records:
        st = r.get("signalSubtype", "unknown")
        if st not in subtype_gtm:
            subtype_gtm[st] = {"total": 0, "nonnull": 0}
        subtype_gtm[st]["total"] += 1
        if r.get("gtmUseCase"):
            subtype_gtm[st]["nonnull"] += 1

    # Culture Praise null rate
    culture_data = subtype_gtm.get("glassdoorHighCulturePraise", {"total": 0, "nonnull": 0})
    culture_null_rate = 1 - (culture_data["nonnull"] / culture_data["total"]) if culture_data["total"] > 0 else 0

    # Risk signals non-null rate
    risk_subtypes = ["glassdoorConsistentLeadershipComplaints", "glassdoorCompensationDissatisfaction",
                     "glassdoorTalentRetentionConcerns"]
    risk_total = sum(subtype_gtm.get(st, {}).get("total", 0) for st in risk_subtypes)
    risk_nonnull = sum(subtype_gtm.get(st, {}).get("nonnull", 0) for st in risk_subtypes)
    risk_nonnull_rate = risk_nonnull / risk_total if risk_total > 0 else 0

    # --- Task 4: Rating summary ---
    rating_summary_count = sum(1 for r in records if r.get("ratingSummary"))
    rating_rate = rating_summary_count / total

    # --- Task 5: Scores ---
    completeness_scores = [r.get("_meta", {}).get("completenessScore", 0) for r in records]
    signal_strengths = [r.get("_meta", {}).get("signalStrength", 0) for r in records]

    # --- Errors ---
    parse_failures = sum(1 for r in records if r.get("_meta", {}).get("errors"))
    error_counts = Counter()
    for r in records:
        for e in r.get("_meta", {}).get("errors", []):
            error_counts[e] += 1
    parse_rate = parse_failures / total

    # --- Build report ---
    report = f"""# Glassdoor v2 Validation Report
Generated: {datetime.now(timezone.utc).isoformat()}
Records: {total}

## Task 1: Signal Subtype Normalization (Python)
Subtype distribution:
"""
    for st, count in subtypes.most_common():
        report += f"  - {st}: {count} ({count/total*100:.1f}%)\n"
    report += f"\nCategory distribution: {dict(categories)}\n"

    report += f"""
## Task 2: Industry Classification
- Industry assigned: {industry_assigned}/{total} -- {check("task2_industry_assigned", industry_rate, 0.85, 0.95)}
- Source breakdown: {dict(industry_sources)}
- Industry distribution (top 10):
"""
    for ind, count in ind_dist.most_common(10):
        report += f"  - {ind}: {count} ({count/total*100:.1f}%)\n"

    report += f"""
## Task 3: GTM Use Case
- Use case non-null: {gtm_nonnull}/{total} -- {check("task3_gtm_nonnull", gtm_rate, 0.30, 0.50)}
- Opportunity category distribution:
"""
    for cat, count in opp_cats.most_common():
        report += f"  - {cat}: {count}\n"

    report += f"\n### GTM non-null rate by signal subtype:\n"
    for st, counts in sorted(subtype_gtm.items(), key=lambda x: -x[1]["total"]):
        rate = counts["nonnull"] / counts["total"] if counts["total"] > 0 else 0
        report += f"  - {st}: {counts['nonnull']}/{counts['total']} ({rate:.0%})\n"

    report += f"\n- Culture Praise null rate: {check('task3_culture_null', culture_null_rate, 0.70, 0.85)}\n"
    report += f"- Risk signals (leadership/comp/retention) non-null rate: {check('task3_risk_nonnull', risk_nonnull_rate, 0.40, 0.65)}\n"

    report += f"""
## Task 4: Rating Summary (Python)
- Rating summaries generated: {rating_summary_count}/{total} -- {check("task4_rating_summary", rating_rate, 0.95, 1.01)}

## Task 5: Deterministic Scores (Python)
- Completeness score avg: {sum(completeness_scores)/len(completeness_scores):.1f} (range: {min(completeness_scores)}-{max(completeness_scores)})
- Signal strength avg: {sum(signal_strengths)/len(signal_strengths):.2f} (range: {min(signal_strengths):.2f}-{max(signal_strengths):.2f})

## Error Summary
- Records with parse failures: {parse_failures}/{total} -- {check("json_parse_failures", parse_rate, 0.0, 0.03)}
- Error breakdown: {dict(error_counts) if error_counts else "none"}

## Threshold Check Summary
"""
    if flags:
        report += "**FLAGS RAISED -- REVIEW BEFORE CONTINUING:**\n"
        for f_item in flags:
            report += f"- [FLAG] {f_item}\n"
    else:
        report += "All metrics within expected ranges. No flags.\n"

    return report, flags


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"Output file not found: {INPUT_FILE}")
        return

    records = load_records()
    print(f"Loaded {len(records)} records")

    report, flags = validate(records)
    print(report)

    report_path = os.path.join(REPORTS_DIR, "validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    if flags:
        print(f"\n*** {len(flags)} FLAG(S) RAISED -- review before continuing ***")


if __name__ == "__main__":
    main()
