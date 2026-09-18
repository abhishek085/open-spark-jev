# Data strategy

One record format (`open_spark_jev/data/corpus.py`), three sources, one loss.

| source | how | gives | use |
|---|---|---|---|
| simulators | `osj simulate` | exact posterior, abstention targets, injected twins | calibration training + benchmark |
| public datasets | `osj public` | hard labels across classification / routing / moderation / scoring / reading | breadth, real language |
| teacher synth | `osj synth --distill` | realistic scenarios with hidden labels + teacher distributions | realism, rare cases |

## Simulators (known posterior)
Domains: routing (5-way choice), security (noul), risk (5-level score), moderation (4-way
choice), incident (4-way choice), game (5-way choice). Each is a naive-Bayes generative model;
`meta.features` and `meta.posterior_max` are stored so you can slice by difficulty.
`--hard-label sample|argmax|latent` controls what the hard label means.

## Teacher generation
Two-stage to keep labels grounded: the teacher writes a state *for a given hidden label*
(so the label is not a guess), then a separate call labels the state blind and returns a
distribution. Disagreement below `min_agreement` flags `meta.suspect` and the record is
excluded from training. Domains: routing, moderation, security, risk, incident, api_trace,
feature_flag, game (`DOMAIN_BRIEFS`).

## Public adapters
ag_news, dair-ai/emotion, PolyAI/banking77 (top-20 intents), lmsys/toxic-chat (noul),
google/boolq (noul with passage as state), Yelp (5-level score). Add one by registering a
generator in `data/public.py`.

## Quality checks to run before training
* label balance per domain (`jq .target.label | sort | uniq -c`)
* teacher agreement histogram; inspect the suspect tail
* duplicate states across records (exact match on `state.content`)
* injected fraction ~10%; abstain fraction ~30%
