from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import PipelineConfig
from ..utils.logging import get_logger

log = get_logger()


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("inf")


@dataclass
class Phase:
    name: str
    role: str  # generator | verifier | semantic_judge | none
    fn: Callable[[], Any]
    when: Callable[[], bool] | None = None  # skip the phase (and never start its server) when this is false


class PhaseScheduler:
    """Runs phases strictly one at a time; a phase's model role must be served before it starts.
    `on_phase_start`/`on_phase_end` are optional hooks (e.g. start/stop a server). Default: log only."""

    def __init__(
        self,
        cfg: PipelineConfig,
        on_phase_start: Callable[[Phase], None] | None = None,
        on_phase_end: Callable[[Phase], None] | None = None,
    ):
        self.cfg = cfg
        self.on_start = on_phase_start or (lambda p: log.info("phase start: %s (role=%s)", p.name, p.role))
        self.on_end = on_phase_end or (lambda p: log.info("phase end: %s", p.name))

    def warn_correlated(self) -> str | None:
        g, v = self.cfg.models.get("generator"), self.cfg.models.get("verifier")
        if g and v and (g.base_url == v.base_url or g.model == v.model):
            msg = "generator and verifier share an endpoint/model: correlated-error risk; records marked lower-confidence"
            log.warning(msg)
            return msg
        return None

    def check_headroom(self) -> None:
        need = self.cfg.scheduler.fail_if_memory_headroom_gb_below
        have = mem_available_gb()
        if have < need:
            raise RuntimeError(f"memory headroom {have:.1f} GB < required {need} GB; stop other model servers first")

    def run(self, phases: list[Phase]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in phases:
            if p.when is not None and not p.when():
                log.info("phase skipped: %s (nothing to do)", p.name)
                continue
            self.on_start(p)
            try:
                out[p.name] = p.fn()
            finally:
                self.on_end(p)
        return out


class CommandServerController:
    """Optional: start/stop a role's server around its phase using `start_cmd` / `stop_cmd` from the model
    config (the pipeline core never embeds serving commands). Only one role's server is kept up."""

    def __init__(self, cfg: PipelineConfig, clients: dict[str, Any]):
        self.cfg, self.clients = cfg, clients
        self.running: str | None = None

    def _sh(self, cmd: str) -> None:
        import subprocess

        subprocess.run(cmd, shell=True, check=False)

    def start(self, phase: Phase) -> None:
        import time

        role = phase.role
        if role not in self.cfg.models or not self.cfg.models[role].start_cmd:
            return
        if self.running and self.running != role and self.cfg.scheduler.unload_between_roles:
            stop = self.cfg.models[self.running].stop_cmd
            if stop:
                log.info("stopping %s server", self.running)
                self._sh(stop)
        self.check_headroom_before(role)
        client = self.clients.get(role)
        if client is not None and hasattr(client, "healthy") and client.healthy()[0]:
            self.running = role
            return
        log.info("starting %s server", role)
        self._sh(self.cfg.models[role].start_cmd or "")
        deadline = time.time() + self.cfg.models[role].startup_timeout_s
        while time.time() < deadline:
            if client is not None and client.healthy()[0]:
                self.running = role
                return
            time.sleep(10)
        raise RuntimeError(f"{role} server did not become healthy in time")

    def check_headroom_before(self, role: str) -> None:
        have = mem_available_gb()
        need = self.cfg.scheduler.fail_if_memory_headroom_gb_below
        if have < need:
            raise RuntimeError(f"memory headroom {have:.1f} GB < {need} GB before starting {role}")

    def stop_all(self) -> None:
        if self.running and self.cfg.models[self.running].stop_cmd and self.cfg.scheduler.unload_between_roles:
            self._sh(self.cfg.models[self.running].stop_cmd or "")
            self.running = None
