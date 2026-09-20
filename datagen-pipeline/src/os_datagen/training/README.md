# Training handoff

See the section **Using this data to train models** in the top-level `README.md` (splits, `os-datagen export-training` formats, open-spark-Jev and TRL examples, calibration gates).

Key rules: train on `train` only; fit calibration on `calibration` only; measure once on `test_locked`; report `challenge` separately; never train numeric probabilities from a teacher's stated confidence; keep the code-rendered policy text in the state.
