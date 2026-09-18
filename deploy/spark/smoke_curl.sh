#!/usr/bin/env bash
# End-to-end check against the gateway (native + Jev-compatible endpoints).
set -euo pipefail
. "$(dirname "$0")/env.sh"
G="http://localhost:$OSJ_GATEWAY_PORT/v1"
curl -s "$G/decide" -H 'Content-Type: application/json' -d '{
  "state": {"content": "Help! My payouts have been failing for 3 days and support is not responding.", "schema_hint": "support ticket"},
  "questions": [
    {"type": "choice", "prompt": "Which queue?", "options": ["billing", "technical", "account", "sales", "abuse"], "allow_abstain": true},
    {"type": "score", "prompt": "How urgent is this?", "levels": ["low", "medium", "high", "critical"]},
    {"type": "noul", "prompt": "The customer is frustrated."}
  ]}' | python3 -m json.tool
curl -s "$G/evaluate" -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "queue": {"type": "choice", "instructions": "Route the ticket", "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}}}
}' | python3 -m json.tool
