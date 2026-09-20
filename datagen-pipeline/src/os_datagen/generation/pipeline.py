from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..datasets import writer
from ..llm.client import LLMClient, LLMRequest
from ..llm.scheduler import Phase, PhaseScheduler
from ..llm.structured_output import StructuredOutputError, parse_model
from ..schemas.common import SPLITS, utc_now
from ..schemas.decision import DatasetRecord, QualityRecord, TruthRecord
from ..schemas.provenance import ProvenanceRecord
from ..schemas.scenario import RenderedState, ScenarioWorld
from ..schemas.validation import JudgeVerdict
from ..taskpacks.base import BaseTaskPack
from ..taskpacks.registry import get_pack
from ..utils.logging import get_logger
from ..utils.paths import git_commit
from ..validation.acceptance import Candidate, Stage
from ..validation.contradiction import check_contradictions
from ..validation.deduplication import dedupe, diversity_select
from ..validation.fact_comparison import check_anchors, comparison_reasons
from ..validation.fact_extraction import extract_facts
from ..validation.leakage import check_leakage
from ..validation.policy_validation import rerun_oracle, validate_record
from ..validation.split_isolation import check_records, check_worlds
from .oracle import solve
from .renderer import PromptLibrary, SurfaceRenderer
from .scenario_sampler import sample_split
from .variants import candidate_id

log = get_logger()


class Pipeline:
    def __init__(self, cfg: PipelineConfig, out_dir: Path, generator: LLMClient, verifier: LLMClient,
                 judge: LLMClient | None = None, dry_run: bool = False, embedder: Any = None,
                 scheduler_hooks: tuple[Any, Any] | None = None, prompts: PromptLibrary | None = None):
        self.cfg, self.out, self.dry_run, self.embedder = cfg, out_dir, dry_run, embedder
        self.gen, self.ver, self.judge = generator, verifier, judge
        self.prompts = prompts or PromptLibrary()
        self.raw_dir = out_dir / "raw"
        self.commit = git_commit()
        self.started = utc_now()
        self.sched = PhaseScheduler(cfg, *(scheduler_hooks or (None, None)))
        self.correlated = self.sched.warn_correlated() if not dry_run else None
        if not dry_run and generator.model_name == verifier.model_name and generator.base_url == verifier.base_url:
            self.correlated = self.correlated or "generator and verifier are the same model"
        self.cands: list[Candidate] = []
        self.packs: dict[str, BaseTaskPack] = {}
        self.truths: dict[str, TruthRecord] = {}
        self.worlds: list[ScenarioWorld] = []
        self.val_rows: list[dict[str, Any]] = []
        self.judge_routed = 0
        self.human_review = 0

    # ---------------------------------------------------------------- phases
    def _sample(self, plan: dict[str, dict[str, int]], seed: int) -> None:
        for name, counts in plan.items():
            pack = get_pack(name)
            self.packs[name] = pack
            start = 1
            for split in SPLITS:
                n = counts.get(split, 0)
                if n <= 0:
                    continue
                ws = sample_split(pack, split, n, seed, start, self.cfg.generation)
                start += len(ws)
                for w in ws:
                    self.truths[w.scenario_id] = solve(pack, w)
                self.worlds += ws
        issues = check_worlds(self.worlds)
        if issues:
            raise RuntimeError(f"split isolation violated at world assignment: {issues}")

    def _render_one(self, renderer: SurfaceRenderer, c: Candidate) -> Candidate:
        pack = self.packs[c.world.task_pack]
        truth = self.truths[c.world.scenario_id]
        req = pack.render_context(c.world, truth)
        rendered, issues, raw = renderer.render(pack, c.world, req, c.candidate_id, c.variant)
        c.raw_path = raw
        c.generator_model = rendered.generator_model if rendered else self.gen.model_name
        if rendered is None:
            c.reject(*issues)
            return c
        c.stage, c.surface = Stage.SCHEMA_OK, pack.finalize_surface(c.world, rendered.surface)
        c.repair_attempts, c.lower_confidence = rendered.repair_attempt_count, rendered.lower_confidence_provenance
        return self._det_gates(pack, c)

    def _det_gates(self, pack: BaseTaskPack, c: Candidate) -> Candidate:
        """Leakage, anchors and structured checks on an already schema-valid surface."""
        assert c.surface is not None
        rendered = RenderedState(surface=c.surface, generator_model=c.generator_model, prompt_version=pack.prompt_version)
        c.state = pack.state_from_surface(c.world, rendered.surface)
        cfg_leak = self._leak_cfg(pack)
        # the scenario family / tags are hidden bookkeeping: their snake_case names must never surface in visible text
        hidden_names = [c.world.scenario_family] + [t for t in c.world.tags if "_" in t]
        cfg_leak["extra_deny"] = list(cfg_leak.get("extra_deny", [])) + [
            r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])" for n in dict.fromkeys(hidden_names) if "_" in n]
        leak = check_leakage(pack, pack.leak_view(c.state), **cfg_leak)
        c.trace["leak_audit"] = leak.matches
        if not leak.ok:
            c.reject(*leak.codes)
            return c
        c.stage = Stage.LEAK_OK
        bad = check_anchors(pack, c.world, c.state) + check_contradictions(pack, c.world, rendered.surface)
        if bad:
            c.reject(*bad)
            return c
        c.stage = Stage.ANCHORS_OK
        return c

    def _leak_cfg(self, pack: BaseTaskPack) -> dict[str, Any]:
        import yaml

        from ..utils.paths import project_path

        p = project_path("configs", "taskpacks", f"{pack.name}.yaml")
        if p.exists():
            leak = (yaml.safe_load(p.read_text()) or {}).get("leakage", {})
            return {"allow_patterns": leak.get("allow_patterns") or [], "extra_deny": leak.get("extra_deny_patterns") or []}
        return {}

    def _generate(self, variants: int) -> None:
        mc = self.cfg.models.get("generator")
        renderer = SurfaceRenderer(self.gen, self.prompts, self.cfg.generation, self.cfg.retention, self.raw_dir,
                                   temperature=mc.temperature if mc else 0.7, top_p=mc.top_p if mc else 0.95,
                                   max_tokens=mc.max_tokens if mc else 1400, lower_confidence=bool(self.correlated))
        cands = [Candidate(candidate_id(w, v), w, v) for w in self.worlds for v in range(variants)]
        conc = mc.concurrency if mc else 2
        with ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
            done = list(ex.map(lambda c: self._render_one(renderer, c), cands))
        self.cands = done
        log.info("generated %d candidates; %d passed deterministic gates", len(done), sum(c.alive for c in done))

    def _verify_one(self, c: Candidate) -> tuple[Candidate, dict[str, Any]]:
        pack = self.packs[c.world.task_pack]
        mc = self.cfg.models.get("verifier")
        info: dict[str, Any] = {}
        try:
            ext, raw, err = extract_facts(pack, c.world, c.state or {}, self.ver, self.prompts,
                                          max_tokens=mc.max_tokens if mc else 900, temperature=mc.temperature if mc else 0.0)
        except Exception as e:  # noqa: BLE001
            c.reject(f"verifier_failed:{type(e).__name__}")
            return c, info
        c.verifier_model = self.ver.model_name
        if ext is None:
            c.reject(err or "verifier_unparseable")
            return c, info
        comp = pack.compare_facts(c.world, ext)
        info = {"extracted": ext.model_dump(), "matched": comp.matched, "mismatched": comp.mismatched,
                "unverifiable": comp.unverifiable}
        if comp.mismatched:
            c.reject(*comparison_reasons(comp))
        elif comp.unverifiable:
            c.trace["needs_judge"] = comparison_reasons(comp)
        else:
            c.stage = Stage.VERIFIED
        return c, info

    def _verify(self) -> None:
        todo = [c for c in self.cands if c.alive]
        mc = self.cfg.models.get("verifier")
        with ThreadPoolExecutor(max_workers=max(1, mc.concurrency if mc else 2)) as ex:
            for c, info in ex.map(self._verify_one, todo):
                c.trace["verification"] = info

    def _needs_judge(self) -> bool:
        return self.judge is not None and self.cfg.validation.use_judge and any(c.alive and c.stage != Stage.VERIFIED for c in self.cands)

    def _judge(self) -> None:
        pending = [c for c in self.cands if c.alive and c.stage != Stage.VERIFIED]
        if not pending:
            return
        vcfg = self.cfg.validation
        cap = max(1, int(vcfg.judge_max_share * max(1, len(self.cands))))
        for i, c in enumerate(pending):
            if self.judge is None or not vcfg.use_judge or i >= cap:
                c.reject(*(c.trace.get("needs_judge") or ["fact_unverifiable"]))
                continue
            self.judge_routed += 1
            pack = self.packs[c.world.task_pack]
            checklist = pack.expected_extraction(c.world)  # hidden-free: facts only, no label
            msg = self.prompts.render("system/semantic_judge_user.j2", {
                "candidate_json": json.dumps(c.state, indent=2, ensure_ascii=False),
                "question_json": json.dumps(pack.question_for(c.world, c.variant), ensure_ascii=False),
                "checklist_json": json.dumps(checklist, ensure_ascii=False),
                "concerns": ", ".join(c.trace.get("needs_judge", [])),
                "schema_json": json.dumps(JudgeVerdict.model_json_schema())})
            mc = self.cfg.models.get("semantic_judge")
            try:
                r = self.judge.chat(LLMRequest(
                    messages=[{"role": "system", "content": self.prompts.render("system/semantic_judge_system.j2", {})},
                              {"role": "user", "content": msg}],
                    temperature=0.0, max_tokens=mc.max_tokens if mc else 900, json_schema=JudgeVerdict.model_json_schema(),
                    metadata={"world": c.world}))
                v = parse_model(r.text, JudgeVerdict)
            except (StructuredOutputError, Exception) as e:  # noqa: BLE001
                c.reject(f"judge_failed:{type(e).__name__}")
                continue
            c.judge_model, c.judged = self.judge.model_name, True
            c.trace["judge"] = v.model_dump()
            if v.verdict == "accept" and v.fact_preservation == "pass" and v.label_leakage == "none" \
                    and v.visible_decision_ambiguity in ("none", "permitted"):
                c.stage = Stage.VERIFIED
            elif v.verdict == "human_review":
                self.human_review += 1
                c.reject("judge:human_review")
            else:
                c.reject("judge:reject", *[f"judge_reason:{x[:60]}" for x in v.reasons[:3]])

    def _make_record(self, c: Candidate) -> DatasetRecord:
        pack = self.packs[c.world.task_pack]
        prov = ProvenanceRecord(
            scenario_id=c.world.scenario_id, scenario_seed=c.world.seed, generator_model=c.generator_model,
            generator_prompt_version=pack.prompt_version, verifier_model=c.verifier_model,
            verifier_prompt_version=pack.verifier_prompt_version, judge_model=c.judge_model,
            repair_attempt_count=c.repair_attempts, lower_confidence_provenance=c.lower_confidence,
            created_at=utc_now(), source_code_commit=self.commit, config_hash=self.cfg.config_hash())
        rendered = RenderedState(surface=c.surface or {}, generator_model=c.generator_model, prompt_version=pack.prompt_version)
        q = QualityRecord(semantic_verification="escalated_passed" if c.judged else "passed", supportability=c.trace.get("support_status", "pending"))
        return pack.build_dataset_record(c.world, self.truths[c.world.scenario_id], rendered, q, prov, variant=c.variant)

    def _support(self) -> None:
        """Blind solvability audit (verifier model, separate prompt). reject: disagreement drops the row; tag: recorded only.
        The challenge split is always tag-only (its rows are hard by design and must not be filtered toward easy)."""
        from ..validation.supportability import audit

        mode = self.cfg.validation.supportability if hasattr(self.cfg.validation, "supportability") else self.cfg.generation.supportability
        todo = [c for c in self.cands if c.stage == Stage.VERIFIED]
        if mode == "off" or not todo:
            for c in todo:
                c.trace["support_status"] = "skipped"
            return
        mc = self.cfg.models.get("verifier")

        def one(c: Candidate) -> None:
            v, reasons = audit(self._make_record(c), self.ver, self.prompts, c.world, max_tokens=900)
            c.trace["support"] = {"verdict": v.model_dump() if v else None, "reasons": reasons}
            pk = self.packs[c.world.task_pack]
            enforce = mode == "reject" and pk.supportability == "reject" and c.world.split != "challenge"
            hard = [r for r in reasons if r.startswith(("supportability:solver_disagrees", "supportability:invalid_choice"))]
            if hard and enforce:  # hedges (also_valid / label_hint / missing_info) are noisy: recorded, never a rejection on their own
                c.reject(*hard)
            else:
                c.trace["support_status"] = "tagged_disagreement" if reasons else "supported"
                if v and v.missing_info:
                    c.trace["support_missing_info"] = v.missing_info

        with ThreadPoolExecutor(max_workers=max(1, mc.concurrency if mc else 2)) as ex:
            list(ex.map(one, todo))

    def _finalize(self) -> dict[str, Any]:
        records: list[DatasetRecord] = []
        by_cand: dict[str, Candidate] = {}
        for c in self.cands:
            if c.stage != Stage.VERIFIED:
                continue
            pack = self.packs[c.world.task_pack]
            truth = self.truths[c.world.scenario_id]
            bad = rerun_oracle(pack, c.world, truth)
            rec = self._make_record(c)
            bad += validate_record(pack, rec)
            if bad:
                c.reject(*bad)
                continue
            records.append(rec)
            by_cand[rec.record_id] = c
        kept, dropped, clusters = dedupe(records, self.cfg.validation.dedupe_near_threshold, self.embedder)
        for r, status, of in dropped:
            by_cand[r.record_id].reject(f"dedupe:{status}:{of}")
        kept_split: dict[str, list[DatasetRecord]] = {s: [r for r in kept if r.split == s] for s in SPLITS}
        tr, cut = diversity_select(kept_split["train"])
        for r in cut:
            by_cand[r.record_id].reject("diversity_cap:easy_share")
        kept_split["train"] = tr
        for s in SPLITS:
            for r in kept_split[s]:
                r.quality.dedupe_status = "unique"
        from .scenario_sampler import family_split

        held = {n: family_split(pk, self.cfg.generation.holdout_max_share)[1] for n, pk in self.packs.items()} if self.cfg.generation.holdout_families else {}
        iso = check_records(kept_split, heldout_families={n: h for n, h in held.items() if h})
        if not iso["ok"]:
            log.error("split isolation violations: %s", iso["violations"][:5])
        for c in self.cands:
            if c.alive and c.stage == Stage.VERIFIED:
                c.stage = Stage.ACCEPTED
        return {"records": kept_split, "clusters": clusters, "isolation": iso, "by_cand": by_cand}

    def load_run(self, run_dir: Path) -> None:
        """Reload worlds, truths and generated surfaces of a previous run (generation is NOT repeated); the
        deterministic gates are re-applied, so verifier/prompt/policy changes can be re-evaluated cheaply."""
        from ..schemas.decision import TruthRecord
        from ..utils.jsonl import read_jsonl

        for row in read_jsonl(run_dir / "scenario_worlds.jsonl"):
            w = ScenarioWorld.model_validate(row["world"])
            self.worlds.append(w)
            self.truths[w.scenario_id] = TruthRecord.model_validate(row["truth"])
            self.packs.setdefault(w.task_pack, get_pack(w.task_pack))
        old_reasons = {r["candidate_id"]: r["reasons"] for r in read_jsonl(run_dir / "rejected.jsonl")}
        wmap = {w.scenario_id: w for w in self.worlds}
        for row in read_jsonl(run_dir / "candidates_raw.jsonl"):
            c = Candidate(row["candidate_id"], wmap[row["scenario_id"]], row["variant"], raw_path=row["raw_generation_path"],
                          generator_model=row["generator_model"], repair_attempts=row["repair_attempt_count"])
            if row["surface"] is None:
                c.reject(*old_reasons.get(c.candidate_id, ["generation_failed"]))
            else:
                pk = self.packs[c.world.task_pack]
                c.stage, c.surface = Stage.SCHEMA_OK, pk.finalize_surface(c.world, row["surface"])
                self._det_gates(pk, c)
            self.cands.append(c)

    def generate_only(self, plan: dict[str, dict[str, int]], seed: int, variants: int = 1) -> dict[str, Any]:
        """Sample + oracle + generate + deterministic gates only; writes a partial run that `reverify` can complete later
        (so the generator model can be unloaded before the verifier model is loaded)."""
        from ..utils.jsonl import write_jsonl

        self.out.mkdir(parents=True, exist_ok=True)
        self.sched.run([Phase("sample_and_oracle", "none", lambda: self._sample(plan, seed)),
                        Phase("generate", "generator", lambda: self._generate(variants))])
        write_jsonl(self.out / "scenario_worlds.jsonl", ({"world": w.model_dump(mode="json"), "truth": self.truths[w.scenario_id].model_dump(mode="json")} for w in self.worlds))
        write_jsonl(self.out / "candidates_raw.jsonl", ({"candidate_id": c.candidate_id, "scenario_id": c.world.scenario_id, "task_pack": c.world.task_pack,
                                                          "split": c.world.split, "variant": c.variant, "stage": c.stage.value, "generator_model": c.generator_model,
                                                          "raw_generation_path": c.raw_path, "repair_attempt_count": c.repair_attempts, "surface": c.surface} for c in self.cands))
        write_jsonl(self.out / "rejected.jsonl", (c.rejection(self.packs[c.world.task_pack].prompt_version) for c in self.cands if not c.alive))
        by: dict[str, Any] = {}
        for n in plan:
            cs = [c for c in self.cands if c.world.task_pack == n]
            reasons: Counter[str] = Counter(":".join(r.split(":")[:2]) for c in cs for r in c.reasons)
            by[n] = {"candidates": len(cs), "passed_deterministic_gates": sum(c.alive for c in cs), "reasons": dict(reasons.most_common(6))}
        return by

    def reverify(self, run_dir: Path, seed: int = 0) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        st: dict[str, Any] = {}
        self.load_run(run_dir)
        self.sched.run([Phase("verify", "verifier", self._verify), Phase("judge", "semantic_judge", self._judge, when=self._needs_judge),
                        Phase("supportability", "verifier", self._support), Phase("finalize", "none", lambda: st.update(self._finalize()))])
        return writer.write_run(self, st, seed=seed, counts={})

    # ---------------------------------------------------------------- driver
    def run(self, pack_names: list[str], counts: dict[str, int] | dict[str, dict[str, int]], seed: int,
            variants: int | None = None) -> dict[str, Any]:
        """counts: {split: n} applied to every pack, or {pack: {split: n}}."""
        plan = counts if counts and isinstance(next(iter(counts.values())), dict) else {n: counts for n in pack_names}
        return self.run_plan(plan, seed, variants)  # type: ignore[arg-type]

    def run_plan(self, plan: dict[str, dict[str, int]], seed: int, variants: int | None = None) -> dict[str, Any]:
        variants = variants or self.cfg.generation.variants_per_world
        self.out.mkdir(parents=True, exist_ok=True)
        st: dict[str, Any] = {}
        counts: dict[str, Any] = plan
        phases = [
            Phase("sample_and_oracle", "none", lambda: self._sample(plan, seed)),
            Phase("generate", "generator", lambda: self._generate(variants)),
            Phase("verify", "verifier", self._verify),
            Phase("judge", "semantic_judge", self._judge, when=self._needs_judge),
            Phase("supportability", "verifier", self._support, when=lambda: any(c.stage == Stage.VERIFIED for c in self.cands)),
            Phase("finalize", "none", lambda: st.update(self._finalize())),
        ]
        self.sched.run(phases)
        return writer.write_run(self, st, seed=seed, counts=counts)
