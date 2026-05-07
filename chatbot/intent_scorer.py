import re

ACTIONS = [
    "SEARCH_TRACK",
    "DISCOVER_MUSIC",
    "SEARCH_AUDIO",
    "ANALYZE_READY",
    "GENERAL_CHAT"
]

def create_score_board():
    return {a: 0.0 for a in ACTIONS}


def add_score(scores, action, value, reason=None):
    scores[action] += value

    if "_reasons" not in scores:
        scores["_reasons"] = []

    scores["_reasons"].append({
        "action": action,
        "score": value,
        "reason": reason
    })


def pick_intent(scores):
    clean_scores = {
        k: v
        for k, v in scores.items()
        if not k.startswith("_")
    }

    best_action = max(clean_scores, key=clean_scores.get)
    best_score = clean_scores[best_action]

    total = sum(clean_scores.values()) + 1e-6

    confidence = best_score / total

    return {
        "intent": best_action,
        "confidence": round(confidence, 3),
        "scores": clean_scores,
        "reasons": scores.get("_reasons", [])
    }