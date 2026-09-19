# Security

## Reporting

Report suspected vulnerabilities by opening a GitHub security advisory on this repository.
Please do not open a public issue for an unpatched vulnerability.

## Using spark-s1 safely

spark-s1 is a decision model. Its output is a probability distribution over a fixed option
set — it cannot emit free text, so it cannot be made to say arbitrary things. That closes one
class of risk and none of the others:

* **The state is untrusted input.** It is rendered inside fences and the model is trained to
  treat it as data, and we measure an injection flip rate (`docs/BENCHMARKS.md`). That rate is
  not zero. An attacker who controls the state can still shift the distribution.
* **Confidence is a model statistic, not a guarantee.** Our own measurements show calibration
  that holds in-distribution and degrades on unfamiliar domains — on one external source the
  model is confidently wrong (`BENCHMARKS.md` M21). Calibrate on your own labelled outcomes
  before thresholding on confidence.
* **Do not put it on consequential paths unattended** — financial, legal, medical, employment,
  safety, or access-control decisions need human review.
* **The playground has no authentication.** `scripts/run_ui.sh` binds localhost by default;
  exposing it beyond a trusted network or tailnet is your responsibility (`docs/UI.md`).
