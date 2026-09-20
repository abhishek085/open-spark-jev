"""Print label/difficulty/family composition and a few rendered prompts from an accepted_*.jsonl file."""
import json
import sys
from collections import Counter

from os_datagen.training.render import render_prompt

rows = [json.loads(line) for line in open(sys.argv[1])]
print(len(rows), "rows")
print("packs:", dict(Counter(r["task_pack"] for r in rows)))
print("difficulty:", dict(Counter(r["quality"]["difficulty"] for r in rows)))
print("truth tiers:", dict(Counter(r["truth"]["label_quality"] for r in rows)))
for r in rows[:3]:
    print("\n" + "=" * 70 + f"\n{r['record_id']} (truth: {r['truth'].get('preferred_option') or r['truth'].get('preferred_level') or r['truth'].get('truth')})\n")
    print(render_prompt(r))
