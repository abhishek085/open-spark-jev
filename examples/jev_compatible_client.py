"""Use the Jev-style wire format against the local gateway (POST /v1/evaluate)."""
import httpx

r = httpx.post("http://localhost:8400/v1/evaluate", json={
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "queue": {"type": "choice", "instructions": "Route the ticket", "criteria": {"billing": "money movement", "technical": "bugs", "sales": "pricing"}},
        "anger": {"type": "score", "instructions": "How angry is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}, timeout=30)
print(r.json())
