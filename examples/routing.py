"""Route a support ticket, score urgency and check a claim, all from one state (one KV prefix)."""
from open_spark_jev.model import MenuScorer
from open_spark_jev.schema import Choice, Noul, Score, State

m = MenuScorer("models/Qwen3-1.7B")
state = State(content="Hi, I was charged twice this month and the invoice PDF is missing. Pro plan.", schema_hint="support ticket")
answers = m.decide(state, [
    Choice(id="queue", prompt="Which support queue?", options=["billing", "technical", "account", "sales", "abuse"], allow_abstain=True),
    Score(id="urgency", prompt="How urgent?", levels=["low", "medium", "high", "critical"]),
    Noul(id="double_charge", prompt="The customer reports a duplicate charge."),
])
for a in answers:
    print(a.id, a.selected, round(a.confidence, 3), a.probability, a.expected_value)
