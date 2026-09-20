"""Sandbox + repeated rollouts for the router: prints the cheapest route whose Wilson lower bound clears the bar."""
import random

from os_datagen.execution.rollouts import summarize
from os_datagen.taskpacks.registry import get_pack

pack = get_pack("harness_next_action_router_v1")
for i in range(200):
    w = pack.sample_world(random.Random(i), "train")
    if w.scenario_family == "exact_local_calculation":
        s = summarize(w.scenario_family, w.facts, ["use_python", "call_small_model", "call_large_model"], 100, i)
        print(s.model_dump_json(indent=2))
        break
