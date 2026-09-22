from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .config import PipelineConfig, load_config
from .datasets.splitter import FILE_FOR_SPLIT, split_counts
from .schemas.common import SPLITS
from .taskpacks.registry import get_pack, pack_names
from .utils.jsonl import read_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False, help="os-datagen: local decision-data factory (open-spark-Jev).")

Config = Annotated[Path | None, typer.Option("--config", help="pipeline yaml (may include models:)")]
Models = Annotated[Path | None, typer.Option("--models", help="models yaml (default: none / dry-run fakes)")]
RunOpt = Annotated[Path, typer.Option("--run", help="run directory, e.g. artifacts/next_action_run_001")]
DatasetOpt = Annotated[Path, typer.Option("--dataset", help="dataset .jsonl")]


def _cfg(config: Path | None, models: Path | None) -> PipelineConfig:
    return load_config(config, models)


def _hooks(cfg: PipelineConfig, gen, ver, judge, dry_run: bool):  # type: ignore[no-untyped-def]
    """Real runs: start each phase's server (start_cmd) and unload the previous role's; dry runs: log only."""
    if dry_run or not any(m.start_cmd for m in cfg.models.values()):
        return None
    from .llm.scheduler import CommandServerController

    ctl = CommandServerController(cfg, {"generator": gen, "verifier": ver, "semantic_judge": judge})
    _CTL.append(ctl)
    return ctl.start, lambda p: None


_CTL: list = []  # type: ignore[type-arg]


def _stop(pipe) -> None:  # type: ignore[no-untyped-def]
    for c in _CTL:
        c.stop_all()
    _CTL.clear()


@app.command("list-packs")
def list_packs() -> None:
    """List task packs and their namespaces."""
    for n in pack_names():
        try:
            p = get_pack(n)
            typer.echo(f"{n:42s} {p.namespace:11s} {p.decision_type:8s} families={len(p.families)}+{len(p.challenge_families)}")
        except ModuleNotFoundError:
            typer.echo(f"{n:42s} (not implemented yet)")


@app.command()
def doctor(config: Config = None, models: Models = None, oracle_samples: int = 30) -> None:
    """Validate config, endpoints, memory headroom, prompts and every pack's oracle."""
    from .generation.renderer import PromptLibrary
    from .generation.scenario_sampler import sample_split
    from .llm.openai_compatible import OpenAICompatibleClient
    from .llm.scheduler import mem_available_gb

    cfg = _cfg(config, models)
    bad = 0
    typer.echo(f"config hash {cfg.config_hash()}; roles: {', '.join(cfg.models) or '(none: dry-run only)'}")
    for role, m in cfg.models.items():
        ok, info = OpenAICompatibleClient(m).healthy()
        typer.echo(f"  [{'ok' if ok else 'DOWN'}] {role}: {m.model} @ {m.base_url} {info if not ok else ''}")
    g, v = cfg.models.get("generator"), cfg.models.get("verifier")
    if g and v and (g.base_url == v.base_url or g.model == v.model):
        typer.echo("  [warn] generator and verifier share an endpoint/model: correlated-error risk")
    typer.echo(f"  memory available: {mem_available_gb():.1f} GB (need >= {cfg.scheduler.fail_if_memory_headroom_gb_below})")
    prompts = PromptLibrary()
    for n in pack_names():
        try:
            p = get_pack(n)
        except ModuleNotFoundError:
            typer.echo(f"  [skip] {n}: not implemented")
            continue
        missing = [t for t in ("surface.j2", "verify.j2") if not (prompts.root / p.prompt_dir / t).exists()]
        try:
            for s in SPLITS:
                sample_split(p, s, oracle_samples, 1, 1, cfg.generation)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            status, bad = f"ORACLE ERROR {e}", bad + 1
        if missing:
            status, bad = status + f" missing prompts {missing}", bad + 1
        typer.echo(f"  [{'ok' if status == 'ok' else 'FAIL'}] {n}: {status}")
    raise typer.Exit(1 if bad else 0)


def _run(cfg: PipelineConfig, packs: list[str], counts_by_pack: dict[str, dict[str, int]], seed: int, out: Path,
         dry_run: bool, variants: int | None, corrupt: float = 0.0, leak: float = 0.0, mismatch: float = 0.0) -> None:
    from .generation.clients import make_clients
    from .generation.pipeline import Pipeline

    gen, ver, judge = make_clients(cfg, dry_run, corrupt, leak, mismatch)
    if dry_run:
        from .config import ModelConfig
        cfg.models.setdefault("generator", ModelConfig(model="fake-generator", temperature=0.0, concurrency=1))
        cfg.models.setdefault("verifier", ModelConfig(model="fake-verifier", concurrency=1))
    pipe = Pipeline(cfg, out, gen, ver, judge, dry_run=dry_run, scheduler_hooks=_hooks(cfg, gen, ver, judge, dry_run))
    try:
        summary = pipe.run(packs, counts_by_pack, seed, variants)
    finally:
        _stop(pipe)
    typer.echo(json.dumps({k: summary[k] for k in ("candidates", "accepted", "rejected", "splits", "namespaces", "split_isolation_ok")}, indent=1))
    for n, b in summary["by_pack"].items():
        typer.echo(f"  {n}: accepted {b['accepted']}/{b['candidates']} ({b['acceptance_rate']:.0%})")
    typer.echo(f"run written to {out}")


@app.command()
def generate(
    task_pack: Annotated[list[str] | None, typer.Option("--task-pack", help="repeatable")] = None,
    all_packs: Annotated[bool, typer.Option("--all")] = False,
    config: Config = None, models: Models = None,
    count: int = 100, seed: int = 42, out: Path = Path("artifacts/run"),
    dry_run: bool = False, variants: int | None = None,
    n_train: int | None = None, n_calibration: int | None = None,
    n_locked_test: int | None = None, n_challenge: int | None = None,
    inject_corrupt: float = 0.0, inject_leak: float = 0.0, inject_mismatch: float = 0.0,
) -> None:
    """Generate, validate and write a dataset for one or more task packs. --count is total worlds per pack."""
    cfg = _cfg(config, models)
    packs = pack_names() if all_packs else (task_pack or [])
    if not packs:
        raise typer.BadParameter("give --task-pack NAME (repeatable) or --all")
    for p in packs:
        get_pack(p)
    sc = split_counts(count, cfg.split_shares)
    for k, v in (("train", n_train), ("calibration", n_calibration), ("locked_test", n_locked_test), ("challenge", n_challenge)):
        if v is not None:
            sc[k] = v
    _run(cfg, packs, sc, seed, out, dry_run, variants, inject_corrupt, inject_leak, inject_mismatch)


@app.command("generate-mixture")
def generate_mixture(
    mixture: Path = Path("configs/mixtures/general_decision_v1.yaml"), config: Config = None, models: Models = None,
    count: int = 1000, seed: int = 42, out: Path = Path("artifacts/mixture"), dry_run: bool = False,
    oversample: float = 1.3,
) -> None:
    """Generate a mixture under composition targets (packs missing for a family are reported as 0%)."""
    cfg = _cfg(config, models)
    spec = yaml.safe_load(mixture.read_text())
    per_pack: dict[str, int] = {}
    for fam, share in spec["targets"].items():
        pk = [p for p in spec["families"].get(fam, []) if p in pack_names()]
        usable = []
        for p in pk:
            try:
                get_pack(p)
                usable.append(p)
            except ModuleNotFoundError:
                typer.echo(f"  [skip] {p} not implemented")
        for p in usable:
            per_pack[p] = per_pack.get(p, 0) + max(1, round(count * oversample * share / len(usable)))
    typer.echo(f"mixture plan (worlds per pack): {per_pack}")
    # run per pack with its own count so the family shares are honored
    from .generation.clients import make_clients
    from .generation.pipeline import Pipeline
    gen, ver, judge = make_clients(cfg, dry_run)
    if dry_run:
        from .config import ModelConfig
        cfg.models.setdefault("generator", ModelConfig(model="fake-generator", concurrency=1))
        cfg.models.setdefault("verifier", ModelConfig(model="fake-verifier", concurrency=1))
    pipe = Pipeline(cfg, out, gen, ver, judge, dry_run=dry_run, scheduler_hooks=_hooks(cfg, gen, ver, judge, dry_run))
    try:
        summary = pipe.run_plan({p: split_counts(n, cfg.split_shares) for p, n in per_pack.items()}, seed)
    finally:
        _stop(pipe)  # unload the last model server (previously left running after generate-mixture)
    typer.echo(json.dumps(summary["composition_vs_target"], indent=1))
    _balance_mixture(out, spec["targets"])


@app.command("generate-code-pack")
def generate_code_pack(pack: str = "harness_tool_call_risk_v1", count: int = 9000, seed: int = 42, out: Path = Path("artifacts/toolrisk")) -> None:
    """Generate a code-only pack (no LLM: grammar + rules establish the label) in the standard run layout."""
    from .generation.code_packs import emit

    if pack != emit.PACK:
        raise typer.BadParameter(f"code-only packs: {emit.PACK}")
    typer.echo(json.dumps(emit.write_run(out, count, seed), indent=1))


@app.command("generate-use-case-packs")
def generate_use_case_packs(count: int = 1500, seed: int = 2027, out: Path = Path("artifacts/jevuse_r1")) -> None:
    """Code-only packs for Jev's structured use cases (entity resolution, fraud, financial triage, control, lead scoring, row validity, semantic grep). `count` is per pack."""
    from .generation.code_packs import emit

    typer.echo(json.dumps(emit.write_use_case_runs(out, count, seed), indent=1))


@app.command("generate-only")
def generate_only(
    task_pack: Annotated[list[str] | None, typer.Option("--task-pack", help="repeatable")] = None,
    all_packs: Annotated[bool, typer.Option("--all")] = False, config: Config = None, models: Models = None,
    count: int = 12, seed: int | None = None, out: Path = Path("artifacts/gen"),
) -> None:
    """Generation phase only (generator model up; no verifier). Finish later with `reverify --run OUT --out NEW`.
    A random seed is used unless --seed is given."""
    import random

    from .generation.clients import make_clients
    from .generation.pipeline import Pipeline

    cfg = _cfg(config, models)
    packs = pack_names() if all_packs else (task_pack or [])
    seed = seed if seed is not None else random.randrange(1, 10**6)
    sc = split_counts(count, cfg.split_shares)
    gen, ver, judge = make_clients(cfg, False)
    pipe = Pipeline(cfg, out, gen, ver, judge)
    by = pipe.generate_only({p: sc for p in packs}, seed)
    typer.echo(f"seed={seed}")
    for n, b in by.items():
        typer.echo(f"  {n}: passed deterministic gates {b['passed_deterministic_gates']}/{b['candidates']}  reasons={b['reasons']}")


@app.command()
def reverify(run: RunOpt, out: Annotated[Path, typer.Option("--out")], config: Config = None, models: Models = None, dry_run: bool = False,
             mixture: Annotated[Path | None, typer.Option("--mixture", help="also trim to this mixture's target composition")] = None) -> None:
    """Re-run verify/judge/finalize on the saved generations of RUN (no regeneration) and write a new run to OUT."""
    from .generation.clients import make_clients
    from .generation.pipeline import Pipeline

    cfg = _cfg(config, models)
    gen, ver, judge = make_clients(cfg, dry_run)
    pipe = Pipeline(cfg, out, gen, ver, judge, dry_run=dry_run, scheduler_hooks=_hooks(cfg, gen, ver, judge, dry_run))
    try:
        summary = pipe.reverify(run)
    finally:
        _stop(pipe)
    for n, b in summary["by_pack"].items():
        typer.echo(f"  {n}: accepted {b['accepted']}/{b['candidates']} ({b['acceptance_rate']:.0%})")
    if mixture:
        _balance_mixture(out, yaml.safe_load(mixture.read_text())["targets"])


@app.command()
def redact(run: RunOpt, delete_raw: bool = False, config: Config = None) -> None:
    """Retention cleanup: redact secret-looking strings in raw generations (or --delete-raw them entirely)."""
    from .utils.redact import redact_run

    cfg = _cfg(config, None)
    typer.echo(json.dumps(redact_run(run, cfg.retention, delete_raw)))


@app.command("export-training")
def export_training_cmd(
    dataset: Annotated[list[Path], typer.Option("--dataset", help="accepted_*.jsonl (repeatable)")],
    out: Annotated[Path, typer.Option("--out")],
    fmt: Annotated[str, typer.Option("--format", help="jev | chat | prompt_completion")] = "jev",
    splits: Annotated[str, typer.Option("--splits", help="comma list; default train only. calibration/locked_test/challenge must not be used for weight training")] = "train",
) -> None:
    """Export records for training: `jev` = open-spark-Jev Record JSONL (menu-scoring trainers), `chat` = messages, `prompt_completion`."""
    from .training.export import export_training

    n = export_training(list(dataset), out, fmt=fmt, allow_splits=set(splits.split(",")))
    typer.echo(f"wrote {n} {fmt} rows ({splits}) to {out}")


@app.command("export-parquet")
def export_parquet(dataset: DatasetOpt, out: Annotated[Path, typer.Option("--out")]) -> None:
    """Materialize a dataset .jsonl as Parquet for analysis (optional; needs pyarrow)."""
    from .datasets.parquet import jsonl_to_parquet

    typer.echo(f"wrote {jsonl_to_parquet(dataset, out)} rows to {out}")


def _balance_mixture(out: Path, targets: dict[str, float]) -> None:
    """Trim the accepted rows to the target composition and write mixture_accepted_*.jsonl + reports/mixture.json."""
    from .datasets.mixture import balance
    from .utils.jsonl import write_jsonl

    rows = {s: list(read_jsonl(out / f)) for s, f in FILE_FOR_SPLIT.items()}
    bal, rep = balance(rows, targets)
    for split, f in FILE_FOR_SPLIT.items():
        write_jsonl(out / f.replace("accepted_", "mixture_accepted_"), bal[split])
    (out / "reports" / "mixture.json").write_text(json.dumps(rep, indent=2))
    typer.echo(f"mixture-balanced rows: {rep['kept_rows']} (binding family {rep['binding_family']}); unfilled families: {rep['unfilled_families']}")


@app.command()
def validate(dataset: DatasetOpt, config: Config = None, models: Models = None,
             prune_out: Annotated[Path | None, typer.Option("--prune-out", help="write only the rows that pass to this .jsonl")] = None) -> None:
    """Re-run schema, leakage and policy gates on an existing JSONL (no LLM calls); optionally write the clean rows."""
    from .schemas.decision import DatasetRecord
    from .utils.jsonl import write_jsonl
    from .validation.leakage import check_leakage
    from .validation.policy_validation import validate_record

    bad, n = 0, 0
    kept: list[dict] = []  # type: ignore[type-arg]
    for row in read_jsonl(dataset):
        n += 1
        before = bad
        try:
            rec = DatasetRecord.model_validate(row)
            pack = get_pack(rec.task_pack)
        except Exception as e:  # noqa: BLE001
            typer.echo(f"row {n}: schema error: {str(e)[:120]}")
            bad += 1
            continue
        reasons = check_leakage(pack, pack.leak_view(rec.decision.state)).codes + validate_record(pack, rec)
        if reasons:
            bad += 1
            typer.echo(f"{rec.record_id}: {reasons}")
        if bad == before:
            kept.append(row)
    if prune_out:
        typer.echo(f"wrote {write_jsonl(prune_out, kept)} clean rows to {prune_out}")
    typer.echo(f"{n} rows checked, {bad} with issues")
    raise typer.Exit(1 if bad else 0)


@app.command()
def report(run: RunOpt) -> None:
    """Print acceptance, rejection, coverage, duplicate and split-isolation reports for a run."""
    rep = run / "reports"
    s = json.loads((rep / "summary.json").read_text())
    typer.echo(f"candidates {s['candidates']}  accepted {s['accepted']}  rejected {s['rejected']}  isolation_ok={s['split_isolation_ok']}")
    typer.echo((rep / "acceptance_by_task.csv").read_text())
    typer.echo("top rejection reasons:\n" + "\n".join((rep / "rejection_reasons.csv").read_text().splitlines()[:15]))
    typer.echo("composition vs target:")
    for f, v in s["composition_vs_target"].items():
        typer.echo(f"  {f:28s} {v['share']:.1%} (target {v['target']:.0%})")
    typer.echo(f"duplicate clusters: {len(json.loads((rep / 'duplicate_clusters.json').read_text()))}; samples: {rep / 'sample_audit.md'}")


@app.command()
def inspect(run: RunOpt, record_id: Annotated[str, typer.Option("--record-id")]) -> None:
    """Show a record (accepted or rejected) with provenance, lineage, validation outcome and raw generation."""
    for name in list(FILE_FOR_SPLIT.values()) + ["rejected.jsonl"]:
        for row in read_jsonl(run / name):
            if row.get("record_id") == record_id or row.get("candidate_id") == record_id:
                typer.echo(f"== {name}\n{json.dumps(row, indent=2, ensure_ascii=False)}")
    for name in ("validation_results.jsonl", "lineage.jsonl"):
        for row in read_jsonl(run / name):
            if record_id in (row.get("candidate_id"), row.get("record_id")):
                typer.echo(f"== {name}\n{json.dumps(row, indent=2, ensure_ascii=False)}")
                if row.get("raw_generation_path"):
                    raw = run / row["raw_generation_path"]
                    if raw.exists():
                        typer.echo(f"== raw generation ({raw})\n{raw.read_text()[:3000]}")


@app.command("option-order")
def option_order(dataset: DatasetOpt, model_endpoint: Annotated[str, typer.Option("--model-endpoint")] = "baseline://uniform",
                 model: Annotated[str, typer.Option("--model")] = "baseline", k: int = 4, out: Path = Path("artifacts/option_order"),
                 limit: int | None = None) -> None:
    """Re-render each record under K option permutations and flag position-sensitive rows (challenge-slice candidates)."""
    from .evaluation.option_order import option_order_check

    s = option_order_check(dataset, out, endpoint=model_endpoint, model=model, k=k, limit=limit)
    typer.echo(json.dumps({k_: v for k_, v in s.items() if k_ != "position_sensitive_record_ids"}, indent=1))


@app.command()
def rollout(task_pack: Annotated[str, typer.Option("--task-pack")], dataset: DatasetOpt,
            routes: str = "use_python,call_small_model,call_large_model",
            runs_per_case: int = 5, out: Path = Path("artifacts/rollout")) -> None:
    """Execute sandbox routes repeatedly and collect empirical outcome data."""
    from .execution.rollouts import run_dataset_rollouts

    n = run_dataset_rollouts(get_pack(task_pack), dataset, routes.split(","), runs_per_case, out)
    typer.echo(f"wrote {n} rollout summaries to {out}")


@app.command()
def evaluate(dataset: DatasetOpt, model_endpoint: Annotated[str, typer.Option("--model-endpoint")],
             model: Annotated[str, typer.Option("--model")], out: Path = Path("artifacts/eval"),
             mode: str = "constrained_generation", limit: int | None = None,
             calibration_dataset: Annotated[Path | None, typer.Option("--calibration-dataset", help="fit temperature on THIS split only")] = None,
             train_dataset: Annotated[Path | None, typer.Option("--train-dataset", help="for baseline://prior")] = None) -> None:
    """Evaluate a candidate decision model on a dataset (per-row predictions + metrics)."""
    from .evaluation.runner import run_evaluation

    m = run_evaluation(dataset, model_endpoint, model, out, mode=mode, limit=limit, calibration_dataset=calibration_dataset, train_path=train_dataset)
    typer.echo(json.dumps(m["overall"], indent=1))


if __name__ == "__main__":
    app()
