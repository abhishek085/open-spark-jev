"""open-spark-Jev: a local, open System One decision model.

Three primitives, one mechanism:

* ``Choice``  - pick one option from a finite list, return a full distribution.
* ``Score``   - place the state on an ordered rubric, return a distribution over levels.
* ``Noul``    - return a calibrated probability that a yes/no claim is true.

All three are answered by a single forward pass of a Qwen3 decoder whose next-token
logits are *restricted* to a small set of label tokens ("menu scoring"). No free-form
generation, no regex parsing, and nothing that a TensorRT-LLM engine cannot serve.
"""

from .schema import (
    Answer,
    Choice,
    DecisionRequest,
    DecisionResponse,
    Noul,
    Question,
    Score,
    State,
)

__version__ = "0.1.0"
__all__ = [
    "classify_tool_call",
    "Answer",
    "Choice",
    "DecisionRequest",
    "DecisionResponse",
    "Noul",
    "Question",
    "Score",
    "State",
]


def classify_tool_call(*args, **kwargs):
    """See :func:`open_spark_jev.gate.classify_tool_call` (imported lazily so `import open_spark_jev` stays light)."""
    from .gate import classify_tool_call as _f

    return _f(*args, **kwargs)
