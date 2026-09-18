"""GPU smoke test: label tokens are single tokens, and the cached-prefix path matches a full
forward pass. Skipped without CUDA + weights."""
import os

import pytest

MODEL = os.environ.get("OSJ_TEST_MODEL", "models/Qwen3-1.7B")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(MODEL), reason="needs GPU + weights")
def test_cache_path_matches_full_forward():
    from open_spark_jev.model import MenuScorer
    from open_spark_jev.schema import Choice, Noul, Score, State

    m = MenuScorer(MODEL)
    s = State(content="[auth] user=u1 failed_attempts=12 geo=tor_exit mfa=failed")
    qs = [
        Choice(prompt="what next?", options=["allow", "block", "challenge"], allow_abstain=True),
        Noul(prompt="This is an attack."),
        Score(prompt="severity?", levels=["low", "mid", "high"]),
    ]
    m.use_state_cache = True
    a, _ = m.label_logits(s, qs)
    m.use_state_cache = False
    b, _ = m.label_logits(s, qs)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=0.15, rtol=0.05), (x, y)
    ans = m.decide(s, qs)
    assert abs(sum(ans[0].probs) - 1) < 1e-6 and ans[1].probability is not None
