"""Deterministic safety-policy layer for proposed shell tool calls.

This is a small, transparent, HEURISTIC layer used by the Decision Lab and the tool gate. It supplements the model; it does not replace
sandboxing, authorization, egress controls, least privilege, code review or human approval, and it is incomplete by design.

Nothing here executes anything: commands are only tokenised and pattern-matched as text.

Fail-closed rules:
* the final action is never less restrictive than the deterministic floor (allow < ask < deny), so a model ``allow`` cannot override a rule;
* sensitive material plus network egress is ``deny``;
* security weakening, mutation, cloud copy/sync and opaque or unsupported input are never auto-allowed (floor ``ask``);
* a model is required for ``allow``: with no model, or with P(allow) below the threshold, the result is ``ask``.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

ORDER = {"allow": 0, "ask": 1, "deny": 2}
SHELL_TOOLS = {"bash", "shell", "sh", "zsh", "terminal", "run_shell", "execute_bash"}

SENSITIVE_PATTERNS = [
    (r"/etc/(shadow|gshadow|sudoers)", "system credential file"),
    (r"\.aws/(credentials|config)", "AWS credentials"),
    (r"\.ssh/|id_(rsa|ed25519|ecdsa)|\.pem\b", "SSH or private key material"),
    (r"kubeconfig|\.kube/config", "kubeconfig"),
    (r"(^|[/\s=@])\.env(\.[\w-]+)?(\s|$|/)", "environment file with secrets"),
    (r"\.(gcloud|azure|docker/config\.json|npmrc|netrc|pgpass)|gcloud/.*credentials|application_default_credentials", "cloud or tool credentials"),
]
CLOUD_COPY = [r"\baws\s+s3\s+(cp|sync|mv)\b", r"\bgsutil\s+(cp|rsync|mv)\b", r"\bgcloud\s+storage\s+(cp|rsync)\b",
              r"\bazcopy\b", r"\baz\s+storage\s+(blob|file)\s+(upload|copy|sync)", r"\brclone\s+(copy|sync|move|copyto)\b", r"\baws\s+s3api\s+put-object\b"]
WEAKENING = [
    (r"(auth|authn|authentication|authorization|authz|rbac|tls|ssl|audit|mtls|verify|verification|secure|security)[\w.\-]*\s*[=: ]\s*(false|0|off|disabled|none)\b", "disables authentication, TLS, RBAC, audit or verification"),
    (r"--(no-verify-ssl|insecure-skip-tls-verify|disable-(auth|tls|rbac|audit)\w*|skip-tls-verify)", "disables a verification or security flag"),
    (r"\b(setenforce\s+0|ufw\s+disable|iptables\s+-F|systemctl\s+(stop|disable)\s+(auditd|firewalld|apparmor))\b", "weakens host security controls"),
    (r"\bchmod\s+(-R\s+)?(0?777|a\+rwx|o\+w)\b", "makes files world-writable"),
]
MUTATION = [
    (r"(^|[\s;&|])(rm|rmdir|unlink|shred|truncate|dd|mkfs\S*)\s", "deletes or overwrites data"),
    (r"\b(chmod|chown|chgrp|setfacl)\s", "changes permissions or ownership"),
    (r"\bkubectl\s+(delete|apply|patch|edit|replace|scale|set|annotate|label|rollout|drain|cordon|taint|create)\b", "mutates a Kubernetes cluster"),
    (r"\bhelm\s+(install|upgrade|uninstall|rollback|delete)\b", "changes a Helm release"),
    (r"\b(terraform|tofu)\s+(apply|destroy|import|taint)\b", "changes infrastructure"),
    (r"\bgit\s+(push|reset\s+--hard|clean|rebase|filter-branch|branch\s+-D)\b", "rewrites or publishes git state"),
    (r"\bdocker\s+(rm|rmi|kill|stop|prune|system\s+prune|volume\s+rm)\b", "removes containers, images or volumes"),
    (r"\bsystemctl\s+(stop|restart|disable|mask)\b", "changes service state"),
    (r"\b(drop|delete|truncate|update|alter|insert)\s+(table|from|database|into)?", "SQL write or destructive statement"),
    (r"\b(pip|npm|apt|apt-get|yum|brew)\s+(install|remove|uninstall|upgrade)\b", "installs or removes software"),
    (r"(^|[^>])>{1,2}\s*[^\s&]", "redirects output into a file"),
    (r"\b(mv|cp)\s", "moves or copies files"),
]
OPAQUE = [
    (r"\b(bash|sh|zsh|dash)\s+-[a-z]*c\b", "opaque shell payload (-c)"),
    (r"\b(eval|source)\s", "eval or source of dynamic content"),
    (r"\$\(|`", "command substitution"),
    (r"\b(python3?|perl|ruby|node)\s+-[ce]\b", "inline interpreter payload"),
    (r"\bbase64\s+(-d|--decode)", "decodes an encoded payload"),
    (r"\|\s*(ba|z)?sh\b", "pipes content into a shell"),
]
EGRESS_CURL_FLAGS = re.compile(r"(^|\s)(-d|--data\S*|-F|--form\S*|-T|--upload-file|--json)(\s|=|$)")


@dataclass
class Match:
    rule: str
    floor: str
    detail: str


@dataclass
class Findings:
    supported: bool
    matches: list[Match] = field(default_factory=list)

    @property
    def floor(self) -> str:
        return max((m.floor for m in self.matches), key=ORDER.get, default="allow")

    def has(self, *rules: str) -> bool:
        return any(m.rule in rules for m in self.matches)


def _first(patterns, text):
    for p in patterns:
        pat, label = p if isinstance(p, tuple) else (p, "")
        m = re.search(pat, text, re.I)
        if m:
            return label, m.group(0).strip()
    return None


def analyze(tool: str, command: str | None) -> Findings:
    """Pattern-match a proposed call. ``command`` is only ever inspected as text."""
    tool = (tool or "").strip().lower()
    if tool not in SHELL_TOOLS:
        return Findings(False, [Match("unsupported_tool", "ask", f"no deterministic rules for tool '{tool or '?'}'; escalating")])
    if not isinstance(command, str) or not command.strip():
        return Findings(False, [Match("unsupported_input", "ask", "empty or non-text command")])
    if len(command) > 8000:
        return Findings(False, [Match("unsupported_input", "ask", "command longer than 8000 characters")])
    out: list[Match] = []
    try:
        shlex.split(command)
    except ValueError:
        out.append(Match("unparsable", "ask", "command could not be tokenised (unbalanced quoting)"))
    sensitive = _first(SENSITIVE_PATTERNS, command)
    if sensitive:
        out.append(Match("sensitive_path", "ask", f"{sensitive[0]}: {sensitive[1]}"))
    egress: list[str] = []
    if re.search(r"\bcurl\b|\bwget\b", command) and (EGRESS_CURL_FLAGS.search(command) or re.search(r"\bwget\b.*--(post-\w+|method=(POST|PUT))", command, re.I)
                                                      or re.search(r"-X\s*(POST|PUT|PATCH)\b", command, re.I)):
        egress.append("HTTP upload (curl/wget with a request body)")
    if re.search(r"\bscp\b", command):
        egress.append("scp transfer")
    if re.search(r"\brsync\b[^|;&]*\s\S+@?\S+:\S*", command) or re.search(r"\brsync\b[^|;&]*\s[\w.\-]+:[^\s]*", command):
        egress.append("rsync to a remote host")
    if re.search(r"\b(nc|ncat|netcat|socat)\b", command) or re.search(r"/dev/(tcp|udp)/", command):
        egress.append("raw network send (nc/socat//dev/tcp)")
    if re.search(r"\b(tar|zip|cat|base64)\b[^|;&]*\|\s*(curl|nc|ncat|ssh|aws|gsutil)\b", command):
        egress.append("archive or content piped to the network")
    for e in egress:
        out.append(Match("network_egress", "ask", e))
    cloud = _first(CLOUD_COPY, command)
    if cloud:
        out.append(Match("cloud_copy", "ask", f"cloud copy or sync: {cloud[1]}"))
    for label_pat in WEAKENING:
        hit = re.search(label_pat[0], command, re.I)
        if hit:
            out.append(Match("security_weakening", "ask", f"{label_pat[1]}: {hit.group(0).strip()}"))
            break
    mut = _first(MUTATION, command)
    if mut:
        out.append(Match("mutation", "ask", f"{mut[0]}: {mut[1]}"))
    opq = _first(OPAQUE, command)
    if opq:
        out.append(Match("opaque", "ask", f"{opq[0]}: {opq[1]}"))
    if any(m.rule == "sensitive_path" for m in out) and any(m.rule in ("network_egress", "cloud_copy") for m in out):
        out.append(Match("sensitive_exfiltration", "deny", "sensitive material combined with network egress or cloud copy"))
    return Findings(True, out)


def resolve(findings: Findings, model_choice: str | None, probabilities: dict[str, float] | None, threshold: float = 0.995) -> tuple[str, list[str]]:
    """Combine the deterministic floor with the model decision. Never less restrictive than the floor. Returns (policy_action, trace)."""
    trace: list[str] = []
    for m in findings.matches:
        trace.append(f"Detected [{m.rule}] {m.detail} (floor: {m.floor})")
    floor = findings.floor
    if model_choice is None or not probabilities:
        trace.append("Model: not available; auto-allow requires a model decision")
        action = floor if floor != "allow" else "ask"
        if floor == "allow":
            trace.append("Final policy action: ask (fail closed without a model)")
        else:
            trace.append(f"Final policy action: {action} (deterministic rules only)")
        return action, trace
    p = {k: float(v) for k, v in probabilities.items()}
    top = max(p, key=p.get)
    trace.append(f"Model: {model_choice} ({p.get(model_choice, 0.0):.3f}); P(allow)={p.get('allow', 0.0):.3f}, threshold {threshold}")
    if model_choice == "allow":
        model_action = "allow" if p.get("allow", 0.0) >= threshold else "ask"
        if model_action == "ask":
            trace.append("Model allow is below the auto-allow threshold: escalate")
    elif model_choice in ORDER:
        model_action = model_choice
    else:
        model_action = "ask"
        trace.append(f"Model returned unknown option '{model_choice}': escalate")
    action = max((model_action, floor), key=ORDER.get)
    if ORDER[floor] > ORDER[model_action]:
        trace.append(f"Deterministic rule overrides the model ({model_action} -> {action})")
    trace.append(f"Final policy action: {action}")
    _ = top
    return action, trace
