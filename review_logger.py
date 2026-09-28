import os
import csv
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "low_confidence_reviews.csv")
JSON_PATH = os.path.join(DATA_DIR, "reviews.json")

CSV_HEADERS = [
    "review_id",
    "timestamp",
    "raw_text",
    "clean_text",
    "predicted_category",
    "category_confidence",
    "predicted_urgency",
    "urgency_confidence",
    "overall_confidence",
    "status",
    "final_category",
    "final_urgency",
    "reviewer_notes",
    "resolved_at"
]


def _init_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

    if not os.path.exists(JSON_PATH):
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


_init_storage()


def log_ticket_prediction(pred_result: Dict[str, Any], auto_logged: bool = False) -> Dict[str, Any]:
    """Logs ticket prediction. If confidence is below threshold or flagged, status is PENDING review;
    otherwise marked as AUTO_ROUTED."""
    _init_storage()

    review_id = f"TCK-{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    needs_review = pred_result.get("needs_human_review", False)
    status = "PENDING_REVIEW" if needs_review else "AUTO_ROUTED"

    record = {
        "review_id": review_id,
        "timestamp": timestamp,
        "raw_text": pred_result.get("raw_text", ""),
        "clean_text": pred_result.get("clean_text", ""),
        "predicted_category": pred_result.get("category", {}).get("prediction", "Unknown"),
        "category_confidence": pred_result.get("category", {}).get("confidence", 0.0),
        "predicted_urgency": pred_result.get("urgency", {}).get("prediction", "Medium"),
        "urgency_confidence": pred_result.get("urgency", {}).get("confidence", 0.0),
        "overall_confidence": pred_result.get("overall_confidence", 0.0),
        "status": status,
        "final_category": pred_result.get("category", {}).get("prediction", "") if not needs_review else "",
        "final_urgency": pred_result.get("urgency", {}).get("prediction", "") if not needs_review else "",
        "reviewer_notes": pred_result.get("review_reason", ""),
        "resolved_at": timestamp if not needs_review else ""
    }

    # Append to CSV
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(record)

    # Append to JSON
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except Exception:
        all_data = []

    all_data.insert(0, record)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    return record


def get_all_reviews() -> List[Dict[str, Any]]:
    """Returns all review records from storage."""
    _init_storage()
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def get_pending_reviews() -> List[Dict[str, Any]]:
    """Returns only low-confidence tickets awaiting human review."""
    reviews = get_all_reviews()
    return [r for r in reviews if r.get("status") == "PENDING_REVIEW"]


def resolve_review(review_id: str, final_category: str, final_urgency: str, reviewer_notes: str = "") -> bool:
    """Updates a pending review with human decisions."""
    _init_storage()
    reviews = get_all_reviews()
    found = False

    resolved_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for r in reviews:
        if r.get("review_id") == review_id:
            r["status"] = "RESOLVED_BY_HUMAN"
            r["final_category"] = final_category
            r["final_urgency"] = final_urgency
            r["reviewer_notes"] = reviewer_notes
            r["resolved_at"] = resolved_time
            found = True
            break

    if not found:
        return False

    # Save JSON
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)

    # Re-write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(reviews)

    return True


def get_review_stats() -> Dict[str, Any]:
    """Returns routing metrics for dashboard."""
    reviews = get_all_reviews()
    total = len(reviews)
    auto_routed = sum(1 for r in reviews if r.get("status") == "AUTO_ROUTED")
    pending = sum(1 for r in reviews if r.get("status") == "PENDING_REVIEW")
    resolved = sum(1 for r in reviews if r.get("status") == "RESOLVED_BY_HUMAN")

    auto_rate = (auto_routed / total * 100) if total > 0 else 0.0
    review_rate = ((pending + resolved) / total * 100) if total > 0 else 0.0

    return {
        "total_tickets": total,
        "auto_routed_count": auto_routed,
        "pending_review_count": pending,
        "human_resolved_count": resolved,
        "auto_route_rate_pct": round(auto_rate, 1),
        "human_review_rate_pct": round(review_rate, 1)
    }
