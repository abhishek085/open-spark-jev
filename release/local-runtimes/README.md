# Local runtimes and community listings: runbook

Everything here creates **new** listings. Nothing modifies an existing Hugging Face repo (`publish_local_runtimes.py` refuses to run if
a target repo already exists). GGUF files are built into `gguf_out/` (git-ignored).

## Build (already done for v6 and v8; for a future release)

```bash
PYTHONPATH=$LLAMA_CPP/gguf-py python $LLAMA_CPP/convert_hf_to_gguf.py checkpoints/<v>-4b --no-mtp --outtype bf16 \
  --outfile gguf_out/spark-s1-4b-<v>-bf16.gguf     # --no-mtp: config declares an MTP layer the merged weights do not have
for q in Q8_0 Q6_K; do $LLAMA_CPP/build/bin/llama-quantize gguf_out/spark-s1-4b-<v>-bf16.gguf gguf_out/spark-s1-4b-<v>-$q.gguf $q; done
```

No Q4_K_M: on v6 it kept accuracy but broke calibration (93% top-1 agreement with HF, probabilities off by up to 0.99), and Ollama /
Docker Model Runner pick it by default when present. v6 parity (200 held-out rows, vs bf16 HF): Q8_0 99.5% / 0.900 acc (HF 0.905),
Q6_K 99.0%, bf16 99.0%; Ollama 0.40.2 Q8_0 99.5%. v8 GGUFs were built the same way but not parity-checked.

## 1. Hugging Face (new repos)

```bash
python scripts/publish_local_runtimes.py --group v6-gguf --dry-run
HF_TOKEN=... python scripts/publish_local_runtimes.py --group v6-gguf        # abhishek085/spark-s1-4b-v6-GGUF
HF_TOKEN=... python scripts/publish_local_runtimes.py --group v8             # abhishek085/spark-s1-4b-v8 (bf16) + -v8-GGUF
```

Add `--private` to inspect before flipping public. Once public, these work with no further submission: `ollama run hf.co/...:Q8_0`,
`llama-server -hf ...:Q8_0`, `docker model pull hf.co/...`, LM Studio / Jan "Use this model".

## 2. ollama.com (`abhishekrai085/spark-s1`)

Needs an ollama.com account with this machine's `~/.ollama/id_ed25519.pub` added under Settings > Ollama keys.

```bash
cd release/local-runtimes/ollama
ollama create abhishekrai085/spark-s1:4b-v6 -f Modelfile.v6 && ollama push abhishekrai085/spark-s1:4b-v6
ollama create abhishekrai085/spark-s1:4b-v8 -f Modelfile.v8 && ollama push abhishekrai085/spark-s1:4b-v8
```

Paste the "How to use" section of the GGUF card into the model's description on ollama.com: it is not a chat model.

## 3. LM Studio Hub

```bash
lms login
cd release/local-runtimes/lmstudio/spark-s1-4b-v6 && lms push
```

`model.yaml` is metadata only and points at the HF GGUF repo, so publish the HF repo first. Unverified: whether LM Studio's server exposes
first-token logprobs; if it does not, LM Studio users can download the model but not use it for decisions, so say so in the description.

## 4. LocalAI gallery

Merged 2026-10-10 as https://github.com/mudler/LocalAI/pull/12609 (`local-ai models install spark-s1-4b-v6`). Upstream changed the description to `logprobs: true` + `top_logprobs: 20`: LocalAI treats a numeric `logprobs` as an on/off flag and then returns only one token.

## 5. Decision Index and Featherless

Drafts in `submissions/`. Decision Index: on hold (decided 2026-10-10). The Decision Index maintainers ask entrants to self-run (`hf-job` runs it on HF hardware) rather than accept run
requests; read the note at the top of `submissions/decision-index.md` before posting.

## Using the builds through the gateway (no torch needed)

```bash
pip install -e ".[serve]"
python -m open_spark_jev.serve.gateway --backend openai --openai-mode completions --upstream http://localhost:8080/v1 --model <dir with calibration.json>
python -m open_spark_jev.serve.gateway --backend ollama --upstream http://localhost:11434 --upstream-model spark-s1-4b-v6:q8_0 --model <dir with calibration.json>
```
