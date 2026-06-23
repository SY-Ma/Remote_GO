from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import signal
import uuid
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit("PyYAML is required. Install it in the local environment first.") from exc

from .adapter import ProjectAdapter
from .run_world import (
    RunWorld,
    display_run_state,
    limit_current_records,
    limit_run_records,
    shorten_run_id,
)


REMOTE_GO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOSTS_CONFIG = REMOTE_GO_ROOT / ".remote_go" / "config.yaml"

COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"
TABLE_SEPARATOR = " "
TABLE_MAX_WIDTHS = {
    "RUN_ID": 36,
    "ID": 26,
    "PROJECT": 18,
    "PROC": 64,
    "NOTE": 58,
    "COMMENT": 72,
    "LOG": 72,
}
TABLE_MIN_WIDTHS = {
    "RUN_ID": 24,
    "ID": 18,
    "MODE": 12,
    "PROC": 32,
    "NOTE": 24,
    "COMMENT": 24,
    "LOG": 32,
}
TABLE_FLEXIBLE_COLUMNS = ("COMMENT", "NOTE", "LOG", "RUN_ID", "ID", "MODE", "PROC")
DEFAULT_RUN_LIMIT = 12


REMOTE_STATUS_CODE = r"""
import json
import os
import subprocess
import time


def run(cmd):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def parse_int(value, default=0):
    try:
        return int(str(value).strip().split()[0])
    except Exception:
        return default


TRACKED_ENV_KEYS = {
    "RUN_ID",
    "RUN_TARGET",
    "RUN_COMMENT",
    "RUN_CHANGE_NOTE",
    "REMOTE_GO_PROJECT",
    "REMOTE_GO_PROJECT_ROOT",
    "REMOTE_GO_PHYSICAL_GPU",
}


def read_env(pid):
    env = {}
    path = f"/proc/{pid}/environ"
    try:
        with open(path, "rb") as file:
            for item in file.read().split(b"\0"):
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                key = key.decode(errors="ignore")
                if key in TRACKED_ENV_KEYS:
                    env[key] = value.decode(errors="ignore")
    except Exception:
        pass
    return env


def infer_project(env):
    if env.get("REMOTE_GO_PROJECT"):
        return env["REMOTE_GO_PROJECT"]
    return ""


def infer_physical_gpu(env):
    return env.get("REMOTE_GO_PHYSICAL_GPU", "")


def read_cwd(pid):
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        return ""


def infer_run_id_from_cwd(cwd):
    project_root = os.environ.get("REMOTE_GO_PROJECT_ROOT", "").rstrip("/")
    if not project_root or not cwd:
        return ""
    prefix = project_root + "/releases/"
    if not cwd.startswith(prefix):
        return ""
    return cwd[len(prefix):].split("/", 1)[0]


def infer_project_from_cwd(cwd):
    project_root = os.environ.get("REMOTE_GO_PROJECT_ROOT", "").rstrip("/")
    if project_root and cwd.startswith(project_root + "/"):
        return os.environ.get("REMOTE_GO_STATUS_PROJECT_LABEL", "")
    return ""


def gpu_lock_held(gpu_id):
    lock_dir = os.environ.get("REMOTE_GO_LOCK_DIR", "")
    if not os.path.isdir(lock_dir):
        return False, ""
    lock_path = os.path.join(lock_dir, f"gpu_{gpu_id}.lock")
    rc, _, err = run(["flock", "-n", lock_path, "true"])
    if rc == 0:
        return False, ""
    return True, err


def read_active_reservation(gpu_id):
    lock_dir = os.environ.get("REMOTE_GO_LOCK_DIR", "")
    if not os.path.isdir(lock_dir):
        return None
    reservation_path = os.path.join(lock_dir, f"gpu_{gpu_id}.reservation.json")
    ttl_seconds = int(os.environ.get("REMOTE_GO_RESERVATION_TTL_SECONDS", "180"))
    try:
        with open(reservation_path, "r") as file:
            payload = json.load(file)
        created_at_epoch = int(payload.get("created_at_epoch", 0))
        age_seconds = int(time.time()) - created_at_epoch
        if age_seconds < ttl_seconds:
            payload["age_seconds"] = age_seconds
            payload["ttl_seconds"] = ttl_seconds
            return payload
    except Exception:
        return None
    return None


gpu_cmd = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]
gpu_rc, gpu_out, gpu_err = run(gpu_cmd)
if gpu_rc != 0:
    print(json.dumps({"error": gpu_err or gpu_out or "nvidia-smi failed", "gpus": []}))
    raise SystemExit(0)

gpus = []
uuid_to_index = {}
for line in gpu_out.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        continue
    index, gpu_uuid, name, mem_used, mem_total, util_gpu = parts[:6]
    gpu = {
        "index": parse_int(index, -1),
        "uuid": gpu_uuid,
        "name": name,
        "memory_used_mib": parse_int(mem_used, -1),
        "memory_total_mib": parse_int(mem_total, -1),
        "utilization_gpu": parse_int(util_gpu, -1),
        "processes": [],
    }
    lock_held, lock_error = gpu_lock_held(gpu["index"])
    gpu["lock_held"] = lock_held
    gpu["lock_error"] = lock_error
    gpu["reservation"] = read_active_reservation(gpu["index"])
    uuid_to_index[gpu_uuid] = gpu["index"]
    gpus.append(gpu)

app_cmd = [
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
]
app_rc, app_out, app_err = run(app_cmd)
if app_rc == 0 and app_out:
    gpu_by_index = {gpu["index"]: gpu for gpu in gpus}
    for line in app_out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid, process_name, used_memory = parts[:4]
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            continue
        ps_rc, ps_out, _ = run(["ps", "-o", "user=", "-p", pid])
        process_user = ps_out.strip() if ps_rc == 0 else ""
        current_user = os.environ.get("USER", "")
        env = read_env(pid) if process_user == current_user else {}
        cwd = read_cwd(pid) if process_user == current_user else ""
        inferred_run_id = infer_run_id_from_cwd(cwd)
        run_id = env.get("RUN_ID", "") or inferred_run_id
        project = infer_project(env) or infer_project_from_cwd(cwd)
        gpu_by_index[gpu_index]["processes"].append({
            "pid": parse_int(pid),
            "user": process_user,
            "is_current_user": process_user == current_user,
            "process_name": process_name,
            "used_memory_mib": parse_int(used_memory),
            "run_id": run_id,
            "run_id_source": "env" if env.get("RUN_ID") else ("cwd" if inferred_run_id else ""),
            "run_target": env.get("RUN_TARGET", ""),
            "project": project,
            "comment": env.get("RUN_COMMENT", ""),
            "change_note": env.get("RUN_CHANGE_NOTE", ""),
            "physical_gpu": infer_physical_gpu(env),
            "cwd": cwd,
        })
elif app_rc != 0:
    for gpu in gpus:
        gpu["process_query_error"] = app_err or app_out or "cannot query compute processes"

print(json.dumps({"error": None, "gpus": gpus}))
"""


@dataclass
class HostConfig:
    name: str
    ssh: str
    idle_mem_threshold_mib: int = 100
    idle_util_threshold_percent: int = 8
    ssh_connect_timeout: int = 8
    gpu_performance_group: str = "unspecified"
    runtime_role: str = "general"
    runtime_note: str = ""


def load_hosts(config_path: Path) -> List[HostConfig]:
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file) or {}
    hosts = []
    host_fields = {field.name for field in fields(HostConfig)}
    for raw_host in raw_config.get("hosts", []):
        hosts.append(HostConfig(**{key: value for key, value in raw_host.items() if key in host_fields}))
    if not hosts:
        raise ValueError(f"No hosts are defined in {config_path}.")
    return hosts


def find_host(hosts: Sequence[HostConfig], name: str) -> HostConfig:
    for host in hosts:
        if host.name == name:
            return host
    raise ValueError(f"Unknown host {name}. Available hosts: {[host.name for host in hosts]}.")


def run_command(cmd: Sequence[str], capture: bool = True, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        check=check,
    )


def run_ssh(host: HostConfig, remote_command: str, capture: bool = True, check: bool = False) -> subprocess.CompletedProcess:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={host.ssh_connect_timeout}",
        host.ssh,
        remote_command,
    ]
    return run_command(cmd, capture=capture, check=check)


def remote_join(*parts: str) -> str:
    clean_parts = []
    for index, part in enumerate(parts):
        if index == 0:
            clean_parts.append(part.rstrip("/"))
        else:
            clean_parts.append(part.strip("/"))
    return posixpath.join(*clean_parts)


def is_under_remote_project_root(adapter: ProjectAdapter, path: str) -> bool:
    root = posixpath.normpath(adapter.remote_project_root)
    target = posixpath.normpath(path)
    return target == root or target.startswith(root + "/")


def is_under_remote_release_dir(adapter: ProjectAdapter, run_id: str, path: str) -> bool:
    if not run_id or not path:
        return False
    release_dir = remote_join(adapter.remote_project_root, "releases", run_id)
    normalized_path = posixpath.normpath(path)
    normalized_release_dir = posixpath.normpath(release_dir)
    return normalized_path == normalized_release_dir or normalized_path.startswith(normalized_release_dir + "/")


def query_project_root_status(adapter: ProjectAdapter, host: HostConfig) -> Dict[str, Any]:
    project_root = shlex.quote(adapter.remote_project_root)
    remote_command = (
        f"if test ! -d {project_root}; then "
        f"echo project_root_missing:{project_root}; exit 66; "
        f"elif test ! -w {project_root}; then "
        f"echo project_root_not_writable:{project_root}; exit 67; "
        f"else echo ok; fi"
    )
    proc = run_ssh(host, remote_command, capture=True, check=False)
    if proc.returncode == 0:
        return {"ok": True, "note": ""}
    note = (proc.stdout or proc.stderr or f"project_root_unavailable:{adapter.remote_project_root}").strip()
    return {"ok": False, "note": note}


def ensure_remote_dirs(adapter: ProjectAdapter, host: HostConfig, dirs: Sequence[str]) -> None:
    project_root_status = query_project_root_status(adapter, host)
    if not project_root_status["ok"]:
        raise RuntimeError(
            f"{host.name} project_root is unavailable; refusing to create parent paths. "
            f"{project_root_status['note']}"
        )
    root = posixpath.normpath(adapter.remote_project_root)
    safe_dirs = []
    for path in dirs:
        if not is_under_remote_project_root(adapter, path):
            raise ValueError(
                f"Refusing to create remote path outside project root: project={adapter.project_id}, "
                f"host={host.name}, path={path}, project_root={adapter.remote_project_root}"
            )
        if posixpath.normpath(path) != root:
            safe_dirs.append(path)
    if not safe_dirs:
        return
    command = "mkdir -p " + " ".join(shlex.quote(path) for path in safe_dirs)
    proc = run_ssh(host, command, capture=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "remote mkdir failed").strip())


def query_host_status(adapter: ProjectAdapter, host: HostConfig) -> Dict[str, Any]:
    remote_lock_dir = remote_join(adapter.remote_project_root, "runs", "gpu_locks")
    remote_command = (
        "REMOTE_GO_LOCK_DIR="
        + shlex.quote(remote_lock_dir)
        + " REMOTE_GO_PROJECT_ROOT="
        + shlex.quote(adapter.remote_project_root)
        + " REMOTE_GO_STATUS_PROJECT_LABEL="
        + shlex.quote(adapter.project_label)
        + " REMOTE_GO_RESERVATION_TTL_SECONDS=180"
        + " python3 -c "
        + shlex.quote(REMOTE_STATUS_CODE)
    )
    proc = run_ssh(host, remote_command, capture=True, check=False)
    if proc.returncode != 0:
        return {
            "project_id": adapter.project_id,
            "project_label": adapter.project_label,
            "host": host.name,
            "ssh": host.ssh,
            "error": (proc.stderr or proc.stdout or "ssh failed").strip(),
            "gpus": [],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"error": proc.stdout.strip() or "invalid status response", "gpus": []}
    rows = [normalize_gpu_status(adapter, host, gpu) for gpu in payload.get("gpus", [])]
    return {
        "project_id": adapter.project_id,
        "project_label": adapter.project_label,
        "host": host.name,
        "ssh": host.ssh,
        "error": payload.get("error"),
        "gpus": rows,
    }


def normalize_gpu_status(adapter: ProjectAdapter, host: HostConfig, gpu: Dict[str, Any]) -> Dict[str, Any]:
    processes = []
    for raw_process in gpu.get("processes", []):
        process = dict(raw_process)
        run_id = str(process.get("run_id") or "")
        belongs_to_project = (
            bool(run_id)
            and process.get("is_current_user")
            and (
                process.get("project") == adapter.project_label
                or is_under_remote_release_dir(adapter, run_id, str(process.get("cwd", "")))
            )
        )
        if run_id and process.get("is_current_user") and not belongs_to_project:
            process["foreign_run_id"] = run_id
            process["run_id"] = ""
            process["run_id_source"] = ""
        processes.append(process)
    tracked_own_processes = [proc for proc in processes if proc.get("run_id")]
    untracked_own_processes = [
        proc for proc in processes if proc.get("is_current_user") and not proc.get("run_id")
    ]
    own_processes = tracked_own_processes + untracked_own_processes
    external_processes = [
        proc for proc in processes if not proc.get("run_id") and not proc.get("is_current_user")
    ]
    memory_used = int(gpu.get("memory_used_mib", 0))
    utilization_gpu = int(gpu.get("utilization_gpu", 0))
    lock_held = bool(gpu.get("lock_held"))
    process_query_error = gpu.get("process_query_error")
    metrics_unknown = memory_used < 0 or utilization_gpu < 0
    reservation = gpu.get("reservation") or {}
    reservation_active = bool(reservation)

    if process_query_error or metrics_unknown:
        state = "unknown"
    elif own_processes and external_processes:
        state = "busy_mixed"
    elif len(own_processes) > 1:
        state = "busy_ours_conflict"
    elif tracked_own_processes:
        state = "busy_ours"
    elif untracked_own_processes:
        state = "busy_ours_untracked"
    elif external_processes:
        state = "busy_external"
    elif reservation_active:
        state = "reserved_ours"
    elif (
        not processes
        and memory_used <= host.idle_mem_threshold_mib
        and utilization_gpu <= host.idle_util_threshold_percent
    ):
        state = "idle"
    elif utilization_gpu > host.idle_util_threshold_percent:
        state = "busy_utilization"
    else:
        state = "busy_memory"

    run_ids = sorted({proc.get("run_id") for proc in tracked_own_processes if proc.get("run_id")})
    if reservation_active and not run_ids and reservation.get("run_id"):
        run_ids = [str(reservation["run_id"])]
    project_names = sorted({proc.get("project") for proc in tracked_own_processes if proc.get("project")})
    if untracked_own_processes:
        project_names.append("untracked_user")
    if reservation_active and not project_names:
        project_names = [adapter.project_label]
    comments = sorted({proc.get("comment", "").strip() for proc in tracked_own_processes if proc.get("comment", "").strip()})
    note_parts = []
    if len(own_processes) > 1:
        note_parts.append(f"multiple_ours:{len(own_processes)}")
    if untracked_own_processes:
        pids = ",".join(str(proc.get("pid", "?")) for proc in untracked_own_processes[:3])
        note_parts.append(f"untracked_ours_pid:{pids}")
    if any(proc.get("run_id_source") == "cwd" for proc in tracked_own_processes):
        note_parts.append("run_id_from_cwd")
    if lock_held:
        note_parts.append("gpu_lock" if processes else "stale_gpu_lock_ignored")
    if comments:
        note_parts.append(f"comment:{comments[0][:40]}")
    if reservation_active:
        ttl = int(reservation.get("ttl_seconds", 0))
        age = int(reservation.get("age_seconds", 0))
        remaining = max(0, ttl - age)
        note_parts.append(f"launch_reservation:{reservation.get('run_id', '')}:{remaining}s")
    if gpu.get("lock_error"):
        note_parts.append(str(gpu["lock_error"]).replace("\n", " ")[:60])
    if process_query_error:
        note_parts.append(str(process_query_error).replace("\n", " ")[:80])
    if metrics_unknown:
        note_parts.append("gpu_metrics_unavailable")
    return {
        "project_id": adapter.project_id,
        "status_scope": adapter.project_label,
        "host": host.name,
        "gpu": int(gpu.get("index", -1)),
        "state": state,
        "name": gpu.get("name", ""),
        "memory_used_mib": memory_used,
        "memory_total_mib": int(gpu.get("memory_total_mib", 0)),
        "utilization_gpu": utilization_gpu,
        "processes": processes,
        "project": ",".join(project_names),
        "run_id": ",".join(run_ids),
        "tmux": f"{adapter.tmux_session}:{adapter.tmux_window}" if run_ids else "",
        "note": "; ".join(note_parts),
    }


def query_all_status(adapters: Sequence[ProjectAdapter], hosts: Sequence[HostConfig], host_name: Optional[str]) -> List[Dict[str, Any]]:
    selected_hosts = [find_host(hosts, host_name)] if host_name else list(hosts)
    payloads = []
    for adapter in adapters:
        for host in selected_hosts:
            payloads.append(query_host_status(adapter, host))
    return payloads


def summarize_processes(processes: Sequence[Dict[str, Any]]) -> str:
    if not processes:
        return "-"
    summaries = []
    for process in processes:
        user = process.get("user") or "?"
        pid = process.get("pid") or "?"
        mem = process.get("used_memory_mib") or 0
        summaries.append(f"{user}:{pid}:{mem}MiB")
    return ",".join(summaries)


def sort_status_processes_for_display(processes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def rank(process: Dict[str, Any]) -> Tuple[int, str]:
        if process.get("run_id"):
            group = 0
        elif process.get("is_current_user"):
            group = 1
        else:
            group = 2
        return group, str(process.get("pid", ""))

    return sorted(processes, key=rank)


def use_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("REMOTE_GO_FORCE_COLOR") or os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        return True
    return sys.stdout.isatty()


def color_text(value: Any, color: str) -> str:
    text = str(value)
    if not use_color():
        return text
    return f"{color}{text}{COLOR_RESET}"


def color_state(state: Any) -> str:
    text = str(state)
    normalized = text.lower()
    if normalized in {"failed", "unknown", "busy_external", "busy_mixed", "busy_utilization", "busy_ours_conflict", "conflict"}:
        return color_text(text, COLOR_RED)
    if normalized in {"running", "idle"}:
        return color_text(text, COLOR_GREEN)
    if normalized in {"busy_ours"}:
        return color_text(text, COLOR_CYAN)
    if normalized in {"completed", "complete", "starting", "reserved_ours", "busy_memory", "busy_ours_untracked"}:
        return color_text(text, COLOR_YELLOW)
    return text


def color_table_cell(header: str, value: Any) -> str:
    if header == "STATE":
        return color_state(value)
    return str(value)


def compact_cell(value: Any, max_width: int) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if max_width <= 0 or len(text) <= max_width:
        return text
    if max_width <= 3:
        return "." * max_width
    return text[: max_width - 3].rstrip() + "..."


def compact_rows_for_width(rows: Sequence[Dict[str, Any]], headers: Sequence[str]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    max_widths = dict(TABLE_MAX_WIDTHS)

    def build_rows() -> List[Dict[str, str]]:
        return [
            {header: compact_cell(row[header], max_widths.get(header, 0)) for header in headers}
            for row in rows
        ]

    def build_widths(compact_rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
        return {header: max(len(header), max(len(str(row[header])) for row in compact_rows)) for header in headers}

    def total_width(widths: Dict[str, int]) -> int:
        return sum(widths[header] for header in headers) + len(TABLE_SEPARATOR) * max(0, len(headers) - 1)

    compact_rows = build_rows()
    widths = build_widths(compact_rows)
    terminal_width = max(80, shutil.get_terminal_size(fallback=(140, 24)).columns)

    while total_width(widths) > terminal_width:
        candidates = [
            header
            for header in TABLE_FLEXIBLE_COLUMNS
            if header in headers and max_widths.get(header, widths[header]) > TABLE_MIN_WIDTHS.get(header, 0)
        ]
        if not candidates:
            break
        widest = max(candidates, key=lambda header: widths.get(header, 0))
        max_widths[widest] = max(TABLE_MIN_WIDTHS[widest], max_widths.get(widest, widths[widest]) - 8)
        compact_rows = build_rows()
        widths = build_widths(compact_rows)

    return compact_rows, widths


def print_status(status_payloads: Sequence[Dict[str, Any]], json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(status_payloads, indent=2))
        return
    rows = []
    for payload in status_payloads:
        if payload.get("error"):
            rows.append({
                "SCOPE": payload.get("project_label", payload.get("project_id", "-")),
                "SERVER": payload["host"],
                "GPU": "-",
                "STATE": "unknown",
                "PROJECT": "-",
                "MEM": "-",
                "UTIL": "-",
                "PROC": "-",
                "RUN_ID": "-",
                "TMUX": "-",
                "NOTE": payload["error"].replace("\n", " ")[:100],
            })
            continue
        for gpu in payload.get("gpus", []):
            processes = sort_status_processes_for_display(gpu.get("processes", []))
            if len(processes) <= 1:
                rows.append({
                    "SCOPE": gpu["status_scope"],
                    "SERVER": gpu["host"],
                    "GPU": str(gpu["gpu"]),
                    "STATE": gpu["state"],
                    "PROJECT": gpu.get("project") or "-",
                    "MEM": f"{gpu['memory_used_mib']}/{gpu['memory_total_mib']}",
                    "UTIL": f"{gpu['utilization_gpu']}%",
                    "PROC": summarize_processes(processes),
                    "RUN_ID": gpu["run_id"] or "-",
                    "TMUX": display_tmux_value(gpu["tmux"]),
                    "NOTE": gpu.get("note", ""),
                })
                continue
            for index, process in enumerate(processes):
                if process.get("run_id"):
                    process_project = process.get("project") or gpu.get("project") or "ours"
                elif process.get("is_current_user"):
                    process_project = "untracked_user"
                else:
                    process_project = "external"
                rows.append({
                    "SCOPE": gpu["status_scope"] if index == 0 else "",
                    "SERVER": gpu["host"] if index == 0 else "",
                    "GPU": str(gpu["gpu"]) if index == 0 else "",
                    "STATE": gpu["state"] if index == 0 else "",
                    "PROJECT": process_project,
                    "MEM": f"{gpu['memory_used_mib']}/{gpu['memory_total_mib']}" if index == 0 else "",
                    "UTIL": f"{gpu['utilization_gpu']}%" if index == 0 else "",
                    "PROC": summarize_processes([process]),
                    "RUN_ID": process.get("run_id") or "-",
                    "TMUX": display_tmux_value(gpu["tmux"]) if process.get("run_id") else "",
                    "NOTE": gpu.get("note", "") if index == 0 else "",
                })
    if not rows:
        print("No GPU status rows.")
        return
    print_table(rows, ["SCOPE", "SERVER", "GPU", "STATE", "PROJECT", "MEM", "UTIL", "PROC", "RUN_ID", "TMUX", "NOTE"], colorize=color_table_cell)


def choose_idle_gpu(status_payloads: Sequence[Dict[str, Any]], host_name: Optional[str], gpu_id: Optional[int]) -> Dict[str, Any]:
    candidates = []
    for payload in status_payloads:
        if payload.get("error"):
            continue
        for gpu in payload.get("gpus", []):
            if host_name is not None and gpu["host"] != host_name:
                continue
            if gpu_id is not None and gpu["gpu"] != gpu_id:
                continue
            if gpu["state"] == "idle":
                candidates.append(gpu)
    if not candidates:
        raise RuntimeError("No completely idle GPU matches the requested constraints.")
    return candidates[0]


def read_registry(path: Path, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if project_id and not record.get("project_id"):
                record["project_id"] = project_id
            records.append(record)
    return records


def append_registry(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def merged_records(adapters: Sequence[ProjectAdapter], include_central: bool = False) -> List[Dict[str, Any]]:
    records = []
    for adapter in adapters:
        records.extend(read_registry(adapter.local_registry, project_id=adapter.project_id))
    deduped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        run_id = record.get("run_id")
        if not run_id:
            continue
        deduped[run_id] = {**deduped.get(run_id, {}), **record}
    return sorted(deduped.values(), key=lambda row: row.get("created_at", ""))


REGISTRY_REQUIRED_FIELDS = (
    "run_id",
    "task",
    "mode",
    "host",
    "gpu",
    "release_dir",
    "log_file",
    "command",
)


def registry_sources(adapters: Sequence[ProjectAdapter]) -> List[Tuple[str, Optional[str], Path, List[Dict[str, Any]]]]:
    sources: List[Tuple[str, Optional[str], Path, List[Dict[str, Any]]]] = []
    for adapter in adapters:
        sources.append(("local", adapter.project_id, adapter.local_registry, read_registry(adapter.local_registry, project_id=adapter.project_id)))
    return sources


def audit_registries(adapters: Sequence[ProjectAdapter]) -> Dict[str, Any]:
    sources = registry_sources(adapters)
    records_by_run_id: Dict[str, List[Tuple[str, Optional[str], Dict[str, Any]]]] = {}
    source_rows = []
    total_records = 0

    for source_type, project_id, path, records in sources:
        missing_field_count = 0
        old_tmux_count = 0
        no_project_id_count = 0
        for record in records:
            total_records += 1
            run_id = record.get("run_id", "")
            if run_id:
                records_by_run_id.setdefault(run_id, []).append((source_type, project_id, record))
            if any(not record.get(field) and record.get(field) != 0 for field in REGISTRY_REQUIRED_FIELDS):
                missing_field_count += 1
            if source_type == "central" and not record.get("project_id"):
                no_project_id_count += 1
            tmux_session = record.get("tmux_session", "")
            tmux_window = record.get("tmux_window", "")
            if tmux_session or tmux_window:
                if f"{tmux_session}:{tmux_window}" != "M:M":
                    old_tmux_count += 1
        source_rows.append({
            "source": source_type,
            "project_id": project_id or "all",
            "path": str(path),
            "records": len(records),
            "missing_required": missing_field_count,
            "central_missing_project_id": no_project_id_count,
            "non_m_m_tmux": old_tmux_count,
        })

    duplicate_run_ids = {
        run_id: [
            {
                "source": source_type,
                "project_id": project_id or item_record.get("project_id", ""),
                "created_at": item_record.get("created_at", ""),
            }
            for source_type, project_id, item_record in items
        ]
        for run_id, items in records_by_run_id.items()
        if len(items) > 1
    }

    central_ids = {
        record.get("run_id")
        for source_type, _, _, records in sources
        if source_type == "central"
        for record in records
        if record.get("run_id")
    }
    local_ids = {
        record.get("run_id")
        for source_type, _, _, records in sources
        if source_type == "local"
        for record in records
        if record.get("run_id")
    }
    return {
        "total_records": total_records,
        "sources": source_rows,
        "duplicate_run_ids": duplicate_run_ids,
        "central_only": sorted(central_ids - local_ids),
        "local_only": sorted(local_ids - central_ids),
        "required_fields": list(REGISTRY_REQUIRED_FIELDS),
    }


def print_registry_audit(audit: Dict[str, Any], json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    rows = []
    for source in audit["sources"]:
        rows.append({
            "SOURCE": source["source"],
            "PROJECT": source["project_id"],
            "RECORDS": str(source["records"]),
            "MISSING": str(source["missing_required"]),
            "NO_PROJECT": str(source["central_missing_project_id"]),
            "NON_M:M": str(source["non_m_m_tmux"]),
            "PATH": source["path"],
        })
    if rows:
        print_table(rows, ["SOURCE", "PROJECT", "RECORDS", "MISSING", "NO_PROJECT", "NON_M:M", "PATH"])
    print(f"duplicate_run_ids: {len(audit['duplicate_run_ids'])}")
    print(f"central_only: {len(audit['central_only'])}")
    print(f"local_only: {len(audit['local_only'])}")


def print_anomaly_report(anomalies: Sequence[Dict[str, Any]], json_output: bool = False) -> None:
    if json_output:
        print(json.dumps({"anomalies": list(anomalies), "count": len(anomalies)}, indent=2, ensure_ascii=False))
        return
    if not anomalies:
        print("Remote_GO refresh: no live identity/GPU invariant violations found.")
        return
    rows = [
        {
            "SEVERITY": item["severity"],
            "TYPE": item["type"],
            "PROJECT": item["project"],
            "SERVER": item["host"],
            "GPU": item["gpu"],
            "STATE": item["state"],
            "RUN_ID": item["run_id"] or "-",
            "NOTE": item["note"],
            "ACTION": item["action"],
        }
        for item in anomalies
    ]
    print_table(rows, ["SEVERITY", "TYPE", "PROJECT", "SERVER", "GPU", "STATE", "RUN_ID", "NOTE", "ACTION"], colorize=color_table_cell)


def fetch_remote_run_status(adapter_by_id: Dict[str, ProjectAdapter], hosts_by_name: Dict[str, HostConfig], record: Dict[str, Any]) -> Dict[str, Any]:
    adapter = adapter_by_id.get(record.get("project_id", ""))
    if adapter is None:
        return {"state": "UNKNOWN", "note": "adapter_not_configured"}
    host = hosts_by_name.get(record.get("host"))
    if host is None:
        return {"state": "UNKNOWN", "note": "host_not_configured"}
    run_id = record.get("run_id", "")
    status_path = remote_join(adapter.remote_project_root, "runs", run_id, "status.json")
    command = f"test -f {shlex.quote(status_path)} && cat {shlex.quote(status_path)}"
    proc = run_ssh(host, command, capture=True, check=False)
    if proc.returncode != 0:
        return {"state": "UNKNOWN", "note": "status_file_missing"}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"state": "UNKNOWN", "note": "invalid_status_json"}


def enrich_records_with_remote_status(records: Sequence[Dict[str, Any]], adapters: Sequence[ProjectAdapter], hosts: Sequence[HostConfig]) -> List[Dict[str, Any]]:
    adapter_by_id = {adapter.project_id: adapter for adapter in adapters}
    hosts_by_name = {host.name: host for host in hosts}
    enriched = []
    for record in records:
        row = dict(record)
        row["remote_status"] = fetch_remote_run_status(adapter_by_id, hosts_by_name, row)
        enriched.append(row)
    return enriched


def query_live_status_for_records(records: Sequence[Dict[str, Any]], adapters: Sequence[ProjectAdapter], hosts: Sequence[HostConfig]) -> List[Dict[str, Any]]:
    adapter_by_id = {adapter.project_id: adapter for adapter in adapters}
    hosts_by_name = {host.name: host for host in hosts}
    seen: set[Tuple[str, str]] = set()
    payloads = []
    for record in records:
        project_id = str(record.get("project_id", ""))
        host_name = str(record.get("host", ""))
        key = (project_id, host_name)
        if key in seen:
            continue
        seen.add(key)
        adapter = adapter_by_id.get(project_id)
        host = hosts_by_name.get(host_name)
        if adapter is None or host is None:
            continue
        payloads.append(query_host_status(adapter, host))
    return payloads


def shorten_time(value: str) -> str:
    if not value:
        return ""
    if "T" not in value or len(value) < 16:
        return value
    return value.replace("T", " ")[5:16]


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def run_duration(remote_status: Dict[str, Any]) -> str:
    started_at = parse_iso_datetime(remote_status.get("started_at"))
    if started_at is None:
        return ""
    state = str(remote_status.get("state", "")).upper()
    finished_at = parse_iso_datetime(remote_status.get("finished_at"))
    updated_at = parse_iso_datetime(remote_status.get("updated_at"))
    if finished_at is not None:
        end_time = finished_at
    elif state == "RUNNING":
        end_time = datetime.now(started_at.tzinfo) if started_at.tzinfo else datetime.now()
    elif updated_at is not None:
        end_time = updated_at
    else:
        return ""
    return format_duration((end_time - started_at).total_seconds())


def display_exit_code(remote_status: Dict[str, Any]) -> str:
    state = str(remote_status.get("state", "")).upper()
    if state not in {"COMPLETED", "FAILED"}:
        return ""
    exit_code = remote_status.get("exit_code")
    return "" if exit_code is None else str(exit_code)


def display_tmux(session: Any, window: Any = "", verbose: bool = False) -> str:
    session_text = str(session or "")
    window_text = str(window or "")
    if not session_text and not window_text:
        return "-"
    if verbose:
        if session_text and window_text and session_text != window_text:
            return f"{session_text}:{window_text}"
        return session_text or window_text
    return session_text or window_text or "-"


def display_tmux_value(value: Any, verbose: bool = False) -> str:
    text = str(value or "")
    if ":" in text:
        session, window = text.split(":", 1)
        return display_tmux(session, window, verbose=verbose)
    return display_tmux(text, "", verbose=verbose)


def print_registry(records: Sequence[Dict[str, Any]], verbose: bool = False) -> None:
    if not records:
        print("No run records found.")
        return
    rows = []
    if verbose:
        headers = ["CREATED", "PROJECT", "RUN_ID", "STATE", "TASK", "MODE", "HOST", "GPU", "EXIT", "DURATION", "UPDATED", "TMUX", "COMMENT", "NOTE", "LOG"]
    else:
        headers = ["CREATED", "PROJECT", "ID", "STATE", "TASK", "MODE", "HOST", "GPU", "EXIT", "DURATION", "TMUX", "COMMENT", "NOTE"]
    for record in records:
        remote_status = record.get("remote_status", {})
        if verbose:
            row = {
                "CREATED": record.get("created_at", ""),
                "PROJECT": record.get("project_id", ""),
                "RUN_ID": record.get("run_id", ""),
                "STATE": display_run_state(remote_status),
                "TASK": record.get("task", ""),
                "MODE": record.get("mode", ""),
                "HOST": record.get("host", ""),
                "GPU": str(record.get("gpu", "")),
                "EXIT": display_exit_code(remote_status),
                "DURATION": run_duration(remote_status),
                "UPDATED": remote_status.get("updated_at", ""),
                "TMUX": display_tmux(record.get("tmux_session", ""), record.get("tmux_window", ""), verbose=True),
                "COMMENT": record.get("comment", ""),
                "NOTE": remote_status.get("note", ""),
                "LOG": record.get("log_file", ""),
            }
        else:
            row = {
                "CREATED": shorten_time(record.get("created_at", "")),
                "PROJECT": record.get("project_id", ""),
                "ID": shorten_run_id(record.get("run_id", "")),
                "STATE": display_run_state(remote_status),
                "TASK": record.get("task", ""),
                "MODE": record.get("mode", ""),
                "HOST": record.get("host", ""),
                "GPU": str(record.get("gpu", "")),
                "EXIT": display_exit_code(remote_status),
                "DURATION": run_duration(remote_status),
                "TMUX": display_tmux(record.get("tmux_session", ""), record.get("tmux_window", ""), verbose=False),
                "COMMENT": record.get("comment", ""),
                "NOTE": remote_status.get("note", ""),
            }
        rows.append(row)
    if not verbose:
        headers = [
            header
            for header in headers
            if header not in {"EXIT", "DURATION", "NOTE"} or any(str(row.get(header, "")).strip() for row in rows)
        ]
    print_table(rows, headers, colorize=color_table_cell)


def print_table(rows: Sequence[Dict[str, Any]], headers: Sequence[str], colorize=None) -> None:
    compact_rows, widths = compact_rows_for_width(rows, headers)
    print(TABLE_SEPARATOR.join(header.ljust(widths[header]) for header in headers))
    print(TABLE_SEPARATOR.join("-" * widths[header] for header in headers))
    for row in compact_rows:
        rendered = []
        for header in headers:
            value = str(row[header])
            rendered_value = colorize(header, value) if colorize is not None else value
            rendered.append(rendered_value + " " * max(0, widths[header] - len(strip_ansi(rendered_value))))
        print(TABLE_SEPARATOR.join(rendered))


def resolve_run_record(records: Sequence[Dict[str, Any]], query: Optional[str]) -> Dict[str, Any]:
    if not records:
        raise ValueError("No run records found.")
    if query is None:
        return records[-1]
    matches = []
    for record in records:
        run_id = str(record.get("run_id", ""))
        if run_id == query or run_id.startswith(query) or run_id.endswith(query) or query in run_id:
            matches.append(record)
    if not matches:
        raise ValueError(f"No run id matches {query}. Use runs to list known runs.")
    if len(matches) > 1:
        match_ids = ", ".join(shorten_run_id(record.get("run_id", "")) for record in matches)
        raise ValueError(f"Run id {query} is ambiguous. Matches: {match_ids}.")
    return matches[0]


def read_remote_log(host: HostConfig, log_file: str, tail_lines: int) -> subprocess.CompletedProcess:
    remote_command = (
        f"if test -f {shlex.quote(log_file)}; then "
        f"tail -n {int(tail_lines)} {shlex.quote(log_file)}; "
        f"else echo 'remote log not found: {shlex.quote(log_file)}' >&2; exit 1; fi"
    )
    return run_ssh(host, remote_command, capture=True, check=False)


def grep_remote_result(host: HostConfig, log_file: str, pattern: str, lines: int) -> subprocess.CompletedProcess:
    remote_command = (
        f"if test -f {shlex.quote(log_file)}; then "
        f"grep -E {shlex.quote(pattern)} {shlex.quote(log_file)} | tail -n {int(lines)}; "
        f"else echo 'remote log not found: {shlex.quote(log_file)}' >&2; exit 1; fi"
    )
    return run_ssh(host, remote_command, capture=True, check=False)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def sync_project(adapter: ProjectAdapter, host: HostConfig, release_dir: str, dry_run: bool = False) -> Dict[str, Any]:
    source = str(adapter.local_root) + "/"
    target = f"{host.ssh}:{release_dir}/"
    cmd = ["rsync", "-az"]
    for pattern in adapter.rsync_exclude_patterns:
        cmd.extend(["--exclude", pattern])
    cmd.extend([source, target])
    if dry_run:
        return {"source": source, "target": target, "command": cmd, "dry_run": True}
    print(f"Syncing {adapter.project_id} to {host.name}:{release_dir} ...")
    proc = run_command(cmd, capture=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"rsync failed with exit code {proc.returncode}.").strip())
    print("Project sync completed.")
    return {"source": source, "target": target, "dry_run": False}


def remote_dir_exists(host: HostConfig, remote_dir: str) -> bool:
    proc = run_ssh(host, f"test -d {shlex.quote(remote_dir)}", capture=True, check=False)
    return proc.returncode == 0


def pull_remote_tree(host: HostConfig, remote_dir: str, local_dir: Path, include_patterns: Sequence[str], dry_run: bool) -> Dict[str, Any]:
    if not dry_run:
        local_dir.mkdir(parents=True, exist_ok=True)
    source = f"{host.ssh}:{remote_dir.rstrip('/')}/"
    cmd = ["rsync", "-az"]
    if dry_run:
        cmd.append("--dry-run")
    for pattern in include_patterns:
        cmd.extend(["--include", pattern])
    cmd.extend(["--exclude", "*", source, str(local_dir) + "/"])
    proc = run_command(cmd, capture=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"rsync pull failed with exit code {proc.returncode}.").strip())
    return {"host": host.name, "remote_dir": remote_dir, "local_dir": str(local_dir), "dry_run": dry_run}


def upload_file(host: HostConfig, local_path: Path, remote_path: str) -> None:
    target = f"{host.ssh}:{shlex.quote(remote_path)}"
    proc = run_command(["rsync", "-az", str(local_path), target], capture=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "file upload failed").strip())


def sync_required_artifacts(adapter: ProjectAdapter, host: HostConfig, release_dir: str, artifacts: Sequence[Tuple[Path, str]]) -> List[str]:
    uploaded = []
    for local_path, relative_path in artifacts:
        remote_path = remote_join(release_dir, relative_path)
        ensure_remote_dirs(adapter, host, [posixpath.dirname(remote_path)])
        upload_file(host, local_path, remote_path)
        uploaded.append(relative_path)
    return uploaded


def shell_array(tokens: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens)


def make_run_id(adapter: ProjectAdapter, task: str, mode: str, name: Optional[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    raw = "_".join(part for part in [timestamp, name, task, mode, suffix] if part)
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw)
    return f"{adapter.project_id}_{safe}" if not safe.startswith(f"{adapter.project_id}_") else safe


def get_git_snapshot(adapter: ProjectAdapter) -> Dict[str, Any]:
    snapshot = {"commit": None, "dirty": None, "status_short": ""}
    commit_proc = run_command(["git", "-C", str(adapter.local_root), "rev-parse", "--short", "HEAD"], capture=True, check=False)
    if commit_proc.returncode == 0:
        snapshot["commit"] = commit_proc.stdout.strip()
    status_proc = run_command(["git", "-C", str(adapter.local_root), "status", "--short"], capture=True, check=False)
    if status_proc.returncode == 0:
        snapshot["status_short"] = status_proc.stdout
        snapshot["dirty"] = bool(status_proc.stdout.strip())
    return snapshot


def read_task_mode(adapter: ProjectAdapter, task: str) -> Optional[str]:
    config_path = adapter.mode_config_path(task)
    if config_path is None:
        return None
    local_config_path = adapter.local_root / config_path
    if not local_config_path.exists():
        return None
    with open(local_config_path, "r") as file:
        config = yaml.safe_load(file) or {}
    mode = config.get("global", {}).get("mode")
    return str(mode) if mode is not None else None


def quote_yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def normalize_yaml_scalar(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def build_mode_config_overlay(adapter: ProjectAdapter, task: str, mode: str) -> Tuple[str, Dict[str, Any]]:
    config_path = adapter.mode_config_path(task)
    if config_path is None:
        raise ValueError(f"Task {task} has no mode config path.")
    local_config_path = adapter.local_root / config_path
    lines = local_config_path.read_text().splitlines(keepends=True)
    global_start = None
    global_end = len(lines)
    for index, line in enumerate(lines):
        if re.match(r"^global\s*:\s*(#.*)?$", line):
            global_start = index
            break
    if global_start is None:
        raise ValueError(f"{config_path} must define a global section before mode can be overridden.")
    for index in range(global_start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith((" ", "\t", "#")):
            global_end = index
            break
    mode_line_pattern = re.compile(r"^(\s*)mode\s*:\s*([^#\r\n]*?)(\s*#.*)?(\r?\n?)$")
    previous_mode = None
    new_mode_value = quote_yaml_string(mode)
    for index in range(global_start + 1, global_end):
        match = mode_line_pattern.match(lines[index])
        if match is None:
            continue
        indent, old_value, comment, newline = match.groups()
        previous_mode = old_value.strip()
        if normalize_yaml_scalar(previous_mode) != mode:
            lines[index] = f"{indent}mode: {new_mode_value}{comment or ''}{newline}"
        break
    else:
        lines.insert(global_start + 1, f"  mode: {new_mode_value}\n")
    return "".join(lines), {
        "config_path": str(config_path),
        "previous_mode": previous_mode,
        "new_mode": mode,
        "scope": "local_config_then_sync",
    }


def update_local_mode_config(adapter: ProjectAdapter, task: str, mode: str) -> Dict[str, Any]:
    config_text, metadata = build_mode_config_overlay(adapter, task, mode)
    config_path = adapter.mode_config_path(task)
    if config_path is None:
        raise ValueError(f"Task {task} has no mode config path.")
    (adapter.local_root / config_path).write_text(config_text)
    return metadata


def make_remote_run_script(
    adapter: ProjectAdapter,
    run_id: str,
    host: HostConfig,
    gpu_id: int,
    release_dir: str,
    remote_run_dir: str,
    log_file: str,
    command_tokens: Sequence[str],
    comment: Optional[str],
    change_note: Optional[str],
) -> str:
    command_with_conda = ["conda", "run", "--no-capture-output", "-n", adapter.conda_env, *command_tokens]
    lock_dir = remote_join(adapter.remote_project_root, "runs", "gpu_locks")
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_ID={shlex.quote(run_id)}
GPU_ID={shlex.quote(str(gpu_id))}
RELEASE_DIR={shlex.quote(release_dir)}
REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
LOG_FILE={shlex.quote(log_file)}
STATUS_FILE="$REMOTE_RUN_DIR/status.json"
IDLE_MEM_THRESHOLD_MIB={shlex.quote(str(host.idle_mem_threshold_mib))}
IDLE_UTIL_THRESHOLD_PERCENT={shlex.quote(str(host.idle_util_threshold_percent))}
LOCK_DIR={shlex.quote(lock_dir)}
RESERVATION_FILE="$LOCK_DIR/gpu_${{GPU_ID}}.reservation.json"
LOCK_FILE="$LOCK_DIR/gpu_${{GPU_ID}}.lock"
COMMAND=({shell_array(command_with_conda)})

write_status() {{
  local state="$1"
  local exit_code="$2"
  python3 - "$STATUS_FILE" "$RUN_ID" "$state" "$exit_code" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
run_id = sys.argv[2]
state = sys.argv[3]
exit_code = int(sys.argv[4])
payload = {{}}
if path.exists():
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {{}}
payload.update({{
    "run_id": run_id,
    "state": state,
    "exit_code": exit_code,
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
}})
if state == "RUNNING" and "started_at" not in payload:
    payload["started_at"] = payload["updated_at"]
if state in {{"COMPLETED", "FAILED"}}:
    payload["finished_at"] = payload["updated_at"]
path.parent.mkdir(parents=True, exist_ok=True)
temp_path = path.with_suffix(path.suffix + ".tmp")
temp_path.write_text(json.dumps(payload, indent=2))
temp_path.replace(path)
PY
}}

finalize_on_exit() {{
  local exit_code="$?"
  rm -f "$RESERVATION_FILE" 2>/dev/null || true
  if [[ "$exit_code" -ne 0 ]]; then
    write_status "FAILED" "$exit_code" || true
  fi
}}
trap finalize_on_exit EXIT

mkdir -p "$LOCK_DIR" "$REMOTE_RUN_DIR" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
write_status "STARTING" 0

init_conda() {{
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  for conda_sh in \\
    "$HOME/.anaconda3/etc/profile.d/conda.sh" \\
    "$HOME/.miniconda3/etc/profile.d/conda.sh" \\
    "$HOME/anaconda3/etc/profile.d/conda.sh" \\
    "$HOME/miniconda3/etc/profile.d/conda.sh" \\
    "/opt/anaconda3/etc/profile.d/conda.sh" \\
    "/opt/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      source "$conda_sh"
      return 0
    fi
  done
  echo "[Remote_GO] conda command is not available in this non-interactive shell."
  return 127
}}

init_conda
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[Remote_GO] GPU $GPU_ID is already reserved by another {adapter.project_label} job."
  write_status "FAILED" 75
  exit 75
fi

python3 - "$GPU_ID" "$IDLE_MEM_THRESHOLD_MIB" "$IDLE_UTIL_THRESHOLD_PERCENT" <<'PY'
import json
import subprocess
import sys

gpu_id = sys.argv[1]
threshold = int(sys.argv[2])
util_threshold = int(sys.argv[3])

def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def parse_int(value):
    try:
        return int(str(value).strip().split()[0])
    except Exception:
        return None

rc, uuid_out, err = run(["nvidia-smi", f"--id={{gpu_id}}", "--query-gpu=uuid", "--format=csv,noheader,nounits"])
if rc != 0:
    print(json.dumps({{"ok": False, "reason": err or "cannot read gpu uuid"}}))
    sys.exit(76)
rc, mem_out, err = run(["nvidia-smi", f"--id={{gpu_id}}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
if rc != 0:
    print(json.dumps({{"ok": False, "reason": err or "cannot read gpu memory"}}))
    sys.exit(76)
rc, util_out, err = run(["nvidia-smi", f"--id={{gpu_id}}", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
if rc != 0:
    print(json.dumps({{"ok": False, "reason": err or "cannot read gpu utilization"}}))
    sys.exit(76)
gpu_uuid = uuid_out.strip()
memory_used = parse_int(mem_out)
utilization_gpu = parse_int(util_out)
if memory_used is None or utilization_gpu is None:
    print(json.dumps({{"ok": False, "reason": "gpu_metrics_unavailable"}}))
    sys.exit(76)
rc, app_out, app_err = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
if rc != 0:
    print(json.dumps({{"ok": False, "reason": app_err or "cannot query compute processes"}}))
    sys.exit(76)
process_count = 0
if app_out:
    for line in app_out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0] == gpu_uuid:
            process_count += 1
if process_count > 0 or memory_used > threshold or utilization_gpu > util_threshold:
    print(json.dumps({{"ok": False, "reason": "gpu_not_idle", "memory_used_mib": memory_used, "utilization_gpu": utilization_gpu, "process_count": process_count}}))
    sys.exit(76)
print(json.dumps({{"ok": True, "gpu_id": gpu_id, "memory_used_mib": memory_used, "utilization_gpu": utilization_gpu, "process_count": process_count}}))
PY

cd "$RELEASE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export RUN_TARGET=remote
export RUN_ID="$RUN_ID"
export RUN_STARTED_AT="$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
export RUN_COMMENT={shlex.quote(comment or "")}
export RUN_CHANGE_NOTE={shlex.quote(change_note or "")}
export REMOTE_GO_PROJECT={shlex.quote(adapter.project_label)}
export REMOTE_GO_PROJECT_ROOT={shlex.quote(adapter.remote_project_root)}
export REMOTE_GO_PHYSICAL_GPU="$GPU_ID"
export PYTHONUNBUFFERED=1

echo "[Remote_GO] run_id=$RUN_ID"
echo "[Remote_GO] project={adapter.project_id} host={host.name} gpu=$GPU_ID release=$RELEASE_DIR"
echo "[Remote_GO] command=${{COMMAND[*]}}"
write_status "RUNNING" 0
set +e
"${{COMMAND[@]}}"
exit_code=$?
set -e
if [[ "$exit_code" -eq 0 ]]; then
  write_status "COMPLETED" "$exit_code"
else
  write_status "FAILED" "$exit_code"
fi
rm -f "$RESERVATION_FILE" 2>/dev/null || true
trap - EXIT
exit "$exit_code"
"""


def make_remote_pane_script(adapter: ProjectAdapter, run_script_path: str, gpu_id: int) -> str:
    pane_title = f"{adapter.display_name}_GPU_{gpu_id}"
    busy_title = f"{pane_title}_BUSY"
    return f"""#!/usr/bin/env bash
set +e

RUN_SCRIPT={shlex.quote(run_script_path)}
PANE_IDLE_TITLE={shlex.quote(pane_title)}
PANE_BUSY_TITLE={shlex.quote(busy_title)}

if [[ -n "${{TMUX_PANE:-}}" ]]; then
  tmux select-pane -t "$TMUX_PANE" -T "$PANE_BUSY_TITLE"
fi

echo "[Remote_GO] pane started at $(date '+%F %T')"
echo "[Remote_GO] running $RUN_SCRIPT"
echo

bash "$RUN_SCRIPT"
exit_code=$?

echo
if [[ "$exit_code" -eq 0 ]]; then
  echo "[Remote_GO] run completed with exit_code=$exit_code at $(date '+%F %T')"
else
  echo "[Remote_GO] run failed with exit_code=$exit_code at $(date '+%F %T')"
fi
echo "[Remote_GO] this tmux pane is kept open for review."
echo "[Remote_GO] type 'exit' or press Ctrl-D in this pane when you want to close it."
echo

if [[ -n "${{TMUX_PANE:-}}" ]]; then
  tmux select-pane -t "$TMUX_PANE" -T "$PANE_IDLE_TITLE"
fi
exec "${{SHELL:-/bin/bash}}" -l
"""


def launch_tmux_pane(adapter: ProjectAdapter, host: HostConfig, run_script_path: str, gpu_id: int, run_id: str) -> None:
    target = f"{adapter.tmux_session}:{adapter.tmux_window}"
    pane_title = f"{adapter.display_name}_GPU_{gpu_id}"
    busy_pane_title = f"{pane_title}_BUSY"
    pane_command = f"bash {shlex.quote(run_script_path)}"
    lock_dir = remote_join(adapter.remote_project_root, "runs", "gpu_locks")
    remote_command = "\n".join([
        "set -e",
        "command -v tmux >/dev/null 2>&1",
        "command -v flock >/dev/null 2>&1",
        f"lock_dir={shlex.quote(lock_dir)}",
        f"launch_guard=$lock_dir/launch_gpu_{gpu_id}.guard",
        f"reservation_file=$lock_dir/gpu_{gpu_id}.reservation.json",
        "reservation_ttl=180",
        "mkdir -p \"$lock_dir\"",
        "if ! mkdir \"$launch_guard\" 2>/dev/null; then",
        f"  echo 'GPU {gpu_id} launch is already being prepared by another request.' >&2",
        "  exit 79",
        "fi",
        "cleanup_launch_guard() {",
        "  rc=\"$?\"",
        "  rmdir \"$launch_guard\" 2>/dev/null || true",
        "  if [ \"$rc\" -ne 0 ]; then rm -f \"$reservation_file\" 2>/dev/null || true; fi",
        "}",
        "trap cleanup_launch_guard EXIT",
        "python3 - \"$reservation_file\" \"$reservation_ttl\" <<'PY'",
        "import json",
        "import sys",
        "import time",
        "from pathlib import Path",
        "path = Path(sys.argv[1])",
        "ttl = int(sys.argv[2])",
        "if path.exists():",
        "    try:",
        "        payload = json.loads(path.read_text())",
        "        age = int(time.time()) - int(payload.get('created_at_epoch', 0))",
        "        if age < ttl:",
        "            print(f\"active launch reservation for {payload.get('run_id', 'unknown')} ({ttl - age}s remaining)\", file=sys.stderr)",
        "            sys.exit(75)",
        "    except Exception:",
        "        pass",
        "sys.exit(0)",
        "PY",
        "python3 - \"$reservation_file\" <<'PY'",
        "import json",
        "import sys",
        "import time",
        "from pathlib import Path",
        f"payload = {{'run_id': {run_id!r}, 'project': {adapter.project_id!r}, 'host': {host.name!r}, 'gpu': {gpu_id}, 'created_at_epoch': int(time.time())}}",
        "Path(sys.argv[1]).write_text(json.dumps(payload, indent=2))",
        "PY",
        f"tmux has-session -t {shlex.quote(adapter.tmux_session)} 2>/dev/null || tmux new-session -d -s {shlex.quote(adapter.tmux_session)} -n {shlex.quote(adapter.tmux_window)}",
        f"tmux list-windows -t {shlex.quote(adapter.tmux_session)} -F '#W' | grep -Fxq {shlex.quote(adapter.tmux_window)} || tmux new-window -t {shlex.quote(adapter.tmux_session + ':')} -n {shlex.quote(adapter.tmux_window)}",
        f"pane_title={shlex.quote(pane_title)}",
        f"busy_pane_title={shlex.quote(busy_pane_title)}",
        f"pane_command={shlex.quote(pane_command)}",
        (
            f"pane_info=$(tmux list-panes -t {shlex.quote(target)} "
            "-F '#{pane_id}\t#{pane_title}\t#{pane_current_command}\t#{pane_dead}' "
            "| awk -F '\\t' -v title=\"$pane_title\" '$2 == title {print; exit}')"
        ),
        "if [ -n \"$pane_info\" ]; then",
        "  IFS=$'\\t' read -r pane_id _pane_title pane_cmd pane_dead <<EOF",
        "$pane_info",
        "EOF",
        "  if [ \"$pane_dead\" = \"1\" ]; then",
        "    pane_info=",
        "  elif ! printf '%s\n' \"$pane_cmd\" | grep -Eq '^(bash|zsh|sh|fish|tcsh|csh)$'; then",
        "    echo \"tmux pane $pane_id for $pane_title is busy with command $pane_cmd\" >&2",
        "    exit 78",
        "  fi",
        "fi",
        "if [ -z \"$pane_info\" ]; then",
        (
            f"  pane_info=$(tmux list-panes -t {shlex.quote(target)} "
            "-F '#{pane_id}\t#{pane_title}\t#{pane_current_command}\t#{pane_dead}' "
            f"| awk -F '\\t' '$2 !~ /^{re.escape(adapter.display_name)}_GPU_/ && $3 ~ /^(bash|zsh|sh|fish|tcsh|csh)$/ && $4 == \"0\" {{print; exit}}')"
        ),
        "  if [ -n \"$pane_info\" ]; then",
        "    IFS=$'\\t' read -r pane_id _pane_title pane_cmd pane_dead <<EOF",
        "$pane_info",
        "EOF",
        "  else",
        f"    pane_id=$(tmux split-window -t {shlex.quote(target)} -v -P -F '#{{pane_id}}')",
        "  fi",
        "fi",
        "tmux select-pane -t \"$pane_id\" -T \"$busy_pane_title\"",
        "tmux send-keys -t \"$pane_id\" \"$pane_command\" C-m",
        f"tmux select-layout -t {shlex.quote(target)} tiled >/dev/null 2>&1 || true",
    ])
    proc = run_ssh(host, remote_command, capture=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "tmux launch failed").strip())


def wait_for_remote_run_status(adapter: ProjectAdapter, host: HostConfig, run_id: str, timeout_seconds: int = 10) -> Optional[Dict[str, Any]]:
    status_path = remote_join(adapter.remote_project_root, "runs", run_id, "status.json")
    attempts = max(1, timeout_seconds * 2)
    remote_command = "\n".join([
        f"status_path={shlex.quote(status_path)}",
        f"attempts={attempts}",
        "i=0",
        "while [ \"$i\" -lt \"$attempts\" ]; do",
        "  if test -f \"$status_path\"; then",
        "    cat \"$status_path\"",
        "    exit 0",
        "  fi",
        "  i=$((i + 1))",
        "  sleep 0.5",
        "done",
        "exit 1",
    ])
    proc = run_ssh(host, remote_command, capture=True, check=False)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def build_metadata(
    adapter: ProjectAdapter,
    run_id: str,
    host: HostConfig,
    gpu_id: int,
    release_dir: str,
    log_file: str,
    command_tokens: Sequence[str],
    task: str,
    mode: str,
    comment: Optional[str],
    change_note: Optional[str],
) -> Dict[str, Any]:
    return {
        "project_id": adapter.project_id,
        "project_label": adapter.project_label,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": task,
        "mode": mode,
        "comment": comment or "",
        "change_note": change_note or "",
        "host": host.name,
        "ssh": host.ssh,
        "gpu": gpu_id,
        "tmux_session": adapter.tmux_session,
        "tmux_window": adapter.tmux_window,
        "release_dir": release_dir,
        "log_file": log_file,
        "command": ["conda", "run", "--no-capture-output", "-n", adapter.conda_env, *command_tokens],
        "local_project_root": str(adapter.local_root),
        "remote_project_root": adapter.remote_project_root,
        "git": get_git_snapshot(adapter),
    }


def registry_has_run_id(adapters: Sequence[ProjectAdapter], run_id: str) -> bool:
    return any(record.get("run_id") == run_id for record in merged_records(adapters, include_central=False))


def find_adoptable_live_process(
    adapter: ProjectAdapter,
    status_payload: Dict[str, Any],
    gpu_id: int,
    run_id: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if status_payload.get("error"):
        raise RuntimeError(f"{status_payload.get('host', '')} status unavailable: {status_payload['error']}")
    matches = [gpu for gpu in status_payload.get("gpus", []) if int(gpu.get("gpu", -1)) == gpu_id]
    if not matches:
        raise ValueError(f"GPU {gpu_id} was not found on {status_payload.get('host', '')}.")
    gpu = matches[0]
    own_tracked = [
        process
        for process in gpu.get("processes", [])
        if process.get("is_current_user") and process.get("run_id")
    ]
    if run_id is not None:
        own_tracked = [process for process in own_tracked if process.get("run_id") == run_id]
    if len(own_tracked) != 1:
        visible = ", ".join(str(process.get("run_id", "")) for process in own_tracked) or "none"
        raise ValueError(
            f"Expected exactly one current-user process with a run_id on GPU {gpu_id}; found {len(own_tracked)} ({visible})."
        )
    process = own_tracked[0]
    actual_run_id = str(process.get("run_id", ""))
    cwd = str(process.get("cwd", ""))
    release_dir = remote_join(adapter.remote_project_root, "releases", actual_run_id)
    normalized_cwd = posixpath.normpath(cwd)
    normalized_release_dir = posixpath.normpath(release_dir)
    if not cwd or not (normalized_cwd == normalized_release_dir or normalized_cwd.startswith(normalized_release_dir + "/")):
        raise ValueError(
            f"Refusing to adopt {actual_run_id}: cwd is not under the expected release dir. "
            f"cwd={cwd}, release_dir={release_dir}"
        )
    if process.get("run_id_source") != "cwd":
        raise ValueError(
            f"Refusing to adopt {actual_run_id}: run_id source is {process.get('run_id_source')!r}, expected cwd."
        )
    if not is_under_remote_project_root(adapter, release_dir):
        raise ValueError(f"Refusing to adopt outside remote project root: {release_dir}")
    return gpu, process


def read_remote_process_argv(host: HostConfig, pid: int) -> List[str]:
    code = r"""
import json
import sys
from pathlib import Path

pid = int(sys.argv[1])
items = [
    item.decode(errors="replace")
    for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if item
]
print(json.dumps(items, ensure_ascii=False))
"""
    proc = run_ssh(host, f"python3 -c {shlex.quote(code)} {int(pid)}", capture=True, check=False)
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
            if isinstance(payload, list):
                return [str(item) for item in payload]
        except json.JSONDecodeError:
            pass
    fallback = run_ssh(host, f"ps -p {int(pid)} -o args=", capture=True, check=False)
    if fallback.returncode != 0 or not fallback.stdout.strip():
        return []
    try:
        return shlex.split(fallback.stdout.strip())
    except ValueError:
        return [fallback.stdout.strip()]


def infer_task_from_command(adapter: ProjectAdapter, command_tokens: Sequence[str]) -> str:
    for task_name, spec in adapter.tasks.items():
        entrypoint_tail = str(spec.entrypoint[-1])
        if any(posixpath.basename(token) == entrypoint_tail for token in command_tokens):
            return task_name
    return "manual"


def infer_mode_from_command(command_tokens: Sequence[str]) -> str:
    for index, token in enumerate(command_tokens):
        if token == "--mode" and index + 1 < len(command_tokens):
            return str(command_tokens[index + 1])
        if token.startswith("--mode="):
            return token.split("=", 1)[1]
    return "manual"


def build_adopted_live_run(
    adapter: ProjectAdapter,
    host: HostConfig,
    gpu: Dict[str, Any],
    process: Dict[str, Any],
    command_tokens: Sequence[str],
    task_override: Optional[str],
    mode_override: Optional[str],
    comment: Optional[str],
    change_note: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    run_id = str(process["run_id"])
    command_tokens = list(command_tokens)
    if not command_tokens and process.get("process_name"):
        command_tokens = [str(process["process_name"])]
    task = task_override or infer_task_from_command(adapter, command_tokens)
    mode = mode_override or infer_mode_from_command(command_tokens)
    release_dir = remote_join(adapter.remote_project_root, "releases", run_id)
    remote_run_dir = remote_join(adapter.remote_project_root, "runs", run_id)
    log_file = remote_join(adapter.remote_project_root, "logs", "manual_adopted", f"{run_id}.log")
    adopted_at = datetime.now().astimezone().isoformat(timespec="seconds")
    adoption_comment = comment or "manual_adopted live process; original launch was not controlled by Remote_GO"
    adoption_change_note = change_note or "Registered existing live process for traceability; no process or code was modified."

    metadata = build_metadata(
        adapter=adapter,
        run_id=run_id,
        host=host,
        gpu_id=int(gpu["gpu"]),
        release_dir=release_dir,
        log_file=log_file,
        command_tokens=command_tokens,
        task=task,
        mode=mode,
        comment=adoption_comment,
        change_note=adoption_change_note,
    )
    metadata["command"] = command_tokens
    metadata["adoption_command_line"] = shell_array(command_tokens)
    metadata["manual_adoption"] = {
        "adopted_at": adopted_at,
        "source": "live_process_cwd",
        "pid": process.get("pid"),
        "cwd": process.get("cwd", ""),
        "run_id_source": process.get("run_id_source", ""),
        "status_state": gpu.get("state", ""),
        "status_note": gpu.get("note", ""),
        "warning": "This run was not launched by Remote_GO; use it as manual/non-standard unless separately validated.",
    }

    status_payload = {
        "run_id": run_id,
        "state": "RUNNING",
        "exit_code": 0,
        "updated_at": adopted_at,
        "adopted_at": adopted_at,
        "started_at_unknown": True,
        "manual_adoption": True,
        "pid": process.get("pid"),
        "cwd": process.get("cwd", ""),
        "command": command_tokens,
        "note": "manual live process adopted into Remote_GO tracking; original launch time is unknown",
    }
    metadata["initial_remote_status"] = status_payload

    adoption_log = "\n".join([
        f"[Remote_GO] adopted live process at {adopted_at}",
        f"[Remote_GO] run_id={run_id}",
        f"[Remote_GO] project={adapter.project_id} host={host.name} gpu={gpu['gpu']} pid={process.get('pid')}",
        f"[Remote_GO] cwd={process.get('cwd', '')}",
        f"[Remote_GO] command={shell_array(command_tokens)}",
        "[Remote_GO] note=manual/non-standard launch; registry/status added after the process was already running.",
        "",
    ])
    return metadata, status_payload, adoption_log


def write_adopted_remote_files(
    adapter: ProjectAdapter,
    host: HostConfig,
    run_id: str,
    metadata: Dict[str, Any],
    status_payload: Dict[str, Any],
    adoption_log: str,
) -> None:
    remote_run_dir = remote_join(adapter.remote_project_root, "runs", run_id)
    log_file = str(metadata["log_file"])
    metadata_path = remote_join(remote_run_dir, "metadata.json")
    status_path = remote_join(remote_run_dir, "status.json")
    ensure_remote_dirs(adapter, host, [remote_run_dir, posixpath.dirname(log_file)])
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        local_metadata = tmp_path / "metadata.json"
        local_status = tmp_path / "status.json"
        local_log = tmp_path / "adoption.log"
        local_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        local_status.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False))
        local_log.write_text(adoption_log)
        upload_file(host, local_metadata, metadata_path)
        upload_file(host, local_status, status_path)
        upload_file(host, local_log, log_file)


def normalize_signal_name(value: str) -> str:
    text = str(value or "TERM").upper()
    if text.isdigit():
        number = int(text)
        for item in signal.Signals:
            if item.value == number:
                return item.name
        raise ValueError(f"Unsupported signal number {number}.")
    if not text.startswith("SIG"):
        text = "SIG" + text
    try:
        signal.Signals[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported signal {value!r}.") from exc
    return text


def run_id_matches_query(run_id: str, query: str) -> bool:
    if not run_id or not query:
        return False
    return run_id == query or run_id.startswith(query) or run_id.endswith(query) or query in run_id


def live_process_belongs_to_adapter(adapter: ProjectAdapter, process: Dict[str, Any], run_id: str) -> bool:
    if not process.get("is_current_user"):
        return False
    if process.get("run_id") != run_id:
        return False
    if process.get("project") == adapter.project_label:
        return True
    return is_under_remote_release_dir(adapter, run_id, str(process.get("cwd", "")))


def find_kill_candidates(
    adapter: ProjectAdapter,
    status_payloads: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    query: str,
    host_name: Optional[str] = None,
    gpu_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    registry_locations = {
        (str(record.get("run_id", "")), str(record.get("host", "")), int(record.get("gpu", -1)))
        for record in records
        if record.get("run_id") and record.get("host") and record.get("gpu") is not None
    }
    candidates: List[Dict[str, Any]] = []
    for payload in status_payloads:
        if payload.get("error"):
            continue
        host = str(payload.get("host", ""))
        if host_name is not None and host != host_name:
            continue
        for gpu in payload.get("gpus", []):
            try:
                gpu_index = int(gpu.get("gpu", -1))
            except (TypeError, ValueError):
                gpu_index = -1
            if gpu_id is not None and gpu_index != gpu_id:
                continue
            for process in gpu.get("processes", []):
                if not process.get("is_current_user"):
                    continue
                run_id = str(process.get("run_id") or "")
                if not run_id_matches_query(run_id, query):
                    continue
                registry_authorized = (run_id, host, gpu_index) in registry_locations
                project_authorized = live_process_belongs_to_adapter(adapter, process, run_id)
                if not registry_authorized and not project_authorized:
                    continue
                candidates.append({
                    "host": host,
                    "gpu": gpu_index,
                    "pid": int(process.get("pid")),
                    "run_id": run_id,
                    "process_name": process.get("process_name", ""),
                    "cwd": process.get("cwd", ""),
                    "project": process.get("project", ""),
                    "registry_authorized": registry_authorized,
                    "project_authorized": project_authorized,
                })
    return candidates


def query_remote_run_processes(adapter: ProjectAdapter, host: HostConfig, run_id: str) -> List[Dict[str, Any]]:
    code = r"""
import json
import os
import sys

expected_run_id = sys.argv[1]
project_root = sys.argv[2].rstrip("/")
project_label = sys.argv[3]
current_uid = os.geteuid()
matches = []

def read_env(pid):
    env = {}
    try:
        with open(f"/proc/{pid}/environ", "rb") as file:
            for item in file.read().split(b"\0"):
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                env[key.decode(errors="ignore")] = value.decode(errors="ignore")
    except Exception:
        return {}
    return env

for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    try:
        proc_stat = os.stat(f"/proc/{pid}")
    except Exception:
        continue
    if proc_stat.st_uid != current_uid:
        continue
    env = read_env(pid)
    if env.get("RUN_ID", "") != expected_run_id:
        continue
    cwd = ""
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        pass
    release_dir = f"{project_root}/releases/{expected_run_id}"
    project_ok = env.get("REMOTE_GO_PROJECT") == project_label
    cwd_ok = bool(cwd) and (cwd == release_dir or cwd.startswith(release_dir + "/"))
    if not project_ok and not cwd_ok:
        continue
    cmdline = ""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as file:
            cmdline = " ".join(item.decode(errors="replace") for item in file.read().split(b"\0") if item)
    except Exception:
        pass
    matches.append({
        "pid": pid,
        "run_id": expected_run_id,
        "cwd": cwd,
        "project": env.get("REMOTE_GO_PROJECT", ""),
        "process_name": cmdline,
        "project_ok": project_ok,
        "cwd_ok": cwd_ok,
    })

print(json.dumps(matches, ensure_ascii=False))
"""
    remote_command = " ".join([
        "python3",
        "-c",
        shlex.quote(code),
        shlex.quote(run_id),
        shlex.quote(adapter.remote_project_root),
        shlex.quote(adapter.project_label),
    ])
    proc = run_ssh(host, remote_command, capture=True, check=False)
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload]


def find_registry_run_ids(records: Sequence[Dict[str, Any]], query: str, host_name: Optional[str], gpu_id: Optional[int]) -> List[str]:
    run_ids = []
    for record in records:
        run_id = str(record.get("run_id") or "")
        if not run_id_matches_query(run_id, query):
            continue
        if host_name is not None and record.get("host") != host_name:
            continue
        if gpu_id is not None:
            try:
                record_gpu = int(record.get("gpu"))
            except (TypeError, ValueError):
                continue
            if record_gpu != gpu_id:
                continue
        run_ids.append(run_id)
    return sorted(set(run_ids))


def kill_remote_process(
    adapter: ProjectAdapter,
    host: HostConfig,
    candidate: Dict[str, Any],
    signal_name: str,
    dry_run: bool,
) -> Dict[str, Any]:
    code = r"""
import json
import os
import signal
import sys

pid = int(sys.argv[1])
expected_run_id = sys.argv[2]
project_root = sys.argv[3].rstrip("/")
project_label = sys.argv[4]
signal_name = sys.argv[5]
dry_run = sys.argv[6] == "1"

def fail(reason):
    print(json.dumps({"ok": False, "reason": reason, "pid": pid, "run_id": expected_run_id}))
    raise SystemExit(1)

try:
    proc_stat = os.stat(f"/proc/{pid}")
except FileNotFoundError:
    fail("process_missing")

if proc_stat.st_uid != os.geteuid():
    fail("not_current_user_process")

env = {}
try:
    with open(f"/proc/{pid}/environ", "rb") as file:
        for item in file.read().split(b"\0"):
            if b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            env[key.decode(errors="ignore")] = value.decode(errors="ignore")
except Exception as exc:
    fail(f"cannot_read_process_env:{exc}")

run_id = env.get("RUN_ID", "")
if run_id != expected_run_id:
    fail(f"run_id_mismatch:{run_id}")

cwd = ""
try:
    cwd = os.readlink(f"/proc/{pid}/cwd")
except Exception:
    pass

release_dir = f"{project_root}/releases/{expected_run_id}"
project_ok = env.get("REMOTE_GO_PROJECT") == project_label
cwd_ok = bool(cwd) and (cwd == release_dir or cwd.startswith(release_dir + "/"))
if not project_ok and not cwd_ok:
    fail("process_not_owned_by_this_remote_go_project")

try:
    sig = getattr(signal, signal_name)
except AttributeError:
    fail(f"unsupported_signal:{signal_name}")

if not dry_run:
    os.kill(pid, sig)

print(json.dumps({
    "ok": True,
    "dry_run": dry_run,
    "pid": pid,
    "run_id": expected_run_id,
    "signal": signal_name,
    "cwd": cwd,
    "project_ok": project_ok,
    "cwd_ok": cwd_ok,
}))
"""
    remote_command = " ".join([
        "python3",
        "-c",
        shlex.quote(code),
        shlex.quote(str(candidate["pid"])),
        shlex.quote(str(candidate["run_id"])),
        shlex.quote(adapter.remote_project_root),
        shlex.quote(adapter.project_label),
        shlex.quote(signal_name),
        "1" if dry_run else "0",
    ])
    proc = run_ssh(host, remote_command, capture=True, check=False)
    text = proc.stdout.strip() or proc.stderr.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {"ok": False, "reason": text or "invalid kill response"}
    payload.update({"host": host.name, "gpu": candidate.get("gpu")})
    if proc.returncode != 0 and payload.get("ok"):
        payload["ok"] = False
    return payload


def command_kill(args: argparse.Namespace, adapter: ProjectAdapter) -> int:
    signal_name = normalize_signal_name(args.signal)
    hosts = load_hosts(args.hosts_config)
    records = merged_records([adapter], include_central=False)
    statuses = query_all_status([adapter], hosts, host_name=args.host)
    candidates = find_kill_candidates(
        adapter=adapter,
        status_payloads=statuses,
        records=records,
        query=args.key,
        host_name=args.host,
        gpu_id=args.gpu,
    )
    if not candidates:
        registry_run_ids = find_registry_run_ids(records, args.key, args.host, args.gpu)
        if len(registry_run_ids) == 1:
            run_id = registry_run_ids[0]
            hosts_by_name = {host.name: host for host in hosts}
            for record in records:
                if record.get("run_id") != run_id:
                    continue
                host_name = str(record.get("host", ""))
                if host_name not in hosts_by_name:
                    continue
                try:
                    record_gpu = int(record.get("gpu", -1))
                except (TypeError, ValueError):
                    record_gpu = -1
                for process in query_remote_run_processes(adapter, hosts_by_name[host_name], run_id):
                    candidates.append({
                        "host": host_name,
                        "gpu": record_gpu,
                        "pid": int(process["pid"]),
                        "run_id": run_id,
                        "process_name": process.get("process_name", ""),
                        "cwd": process.get("cwd", ""),
                        "project": process.get("project", ""),
                        "registry_authorized": True,
                        "project_authorized": True,
                    })
    if not candidates:
        print_status(statuses, json_output=False)
        raise ValueError(
            f"No killable current-user Remote_GO process matches {args.key!r}. "
            "Use go status or go runs to copy a visible run id."
        )
    unique_run_ids = sorted({candidate["run_id"] for candidate in candidates})
    if len(unique_run_ids) > 1:
        visible = ", ".join(shorten_run_id(run_id) for run_id in unique_run_ids)
        raise ValueError(f"Kill key {args.key!r} is ambiguous across run ids: {visible}. Use a longer key.")
    location_keys = sorted({(item["host"], item["gpu"]) for item in candidates})
    if len(location_keys) > 1 and not args.all:
        locations = ", ".join(f"{item['host']}:{item['gpu']}:{item['pid']}" for item in candidates)
        raise ValueError(
            f"Run {unique_run_ids[0]} has live process candidates in multiple locations ({locations}). "
            "Use --host/--gpu to narrow it, or --all to signal all matching current-user project processes."
        )
    hosts_by_name = {host.name: host for host in hosts}
    results = []
    for candidate in candidates:
        host = hosts_by_name[candidate["host"]]
        results.append(kill_remote_process(adapter, host, candidate, signal_name, dry_run=args.dry_run))
    print(json.dumps({"signal": signal_name, "dry_run": args.dry_run, "results": results}, indent=2))
    if any(not result.get("ok") for result in results):
        return 1
    return 0


def command_status(args: argparse.Namespace, adapters: Sequence[ProjectAdapter]) -> int:
    hosts = load_hosts(args.hosts_config)
    statuses = query_all_status(adapters, hosts, host_name=args.host)
    print_status(statuses, json_output=args.json)
    return 0


def command_runs(args: argparse.Namespace, adapters: Sequence[ProjectAdapter]) -> int:
    all_records = merged_records(adapters, include_central=False)
    records = all_records
    show_history = bool(getattr(args, "history", False) or getattr(args, "all_records", False) or args.local_only)
    record_limit = args.limit if show_history else max(args.limit * 2, args.limit)
    records = limit_run_records(records, record_limit, getattr(args, "all_records", False))
    if not args.local_only and (records or not show_history):
        hosts = load_hosts(args.hosts_config)
        if not show_history:
            status_payloads = query_all_status(adapters, hosts, host_name=None)
            live_run_ids = {
                str(run_id)
                for payload in status_payloads
                if not payload.get("error")
                for gpu in payload.get("gpus", [])
                for run_id in [proc.get("run_id") for proc in gpu.get("processes", [])]
                if run_id
            }
            records_by_id = {str(record.get("run_id", "")): record for record in records if record.get("run_id")}
            for record in all_records:
                run_id = str(record.get("run_id", ""))
                if run_id and run_id in live_run_ids:
                    records_by_id[run_id] = record
            records = list(records_by_id.values())
            records = enrich_records_with_remote_status(records, adapters, hosts)
            known_run_ids = {str(record.get("run_id")) for record in all_records if record.get("run_id")}
            world = RunWorld.from_sources(records, status_payloads, known_run_ids=known_run_ids)
            records = limit_current_records(world.current_records(), args.limit)
        else:
            records = enrich_records_with_remote_status(records, adapters, hosts)
    if getattr(args, "json", False):
        print(json.dumps({
            "count": len(records),
            "limit": args.limit,
            "records": records,
        }, indent=2, ensure_ascii=False))
        return 0
    print_registry(records, verbose=args.verbose)
    return 0


def command_registry_audit(args: argparse.Namespace, adapters: Sequence[ProjectAdapter]) -> int:
    audit = audit_registries(adapters)
    print_registry_audit(audit, json_output=args.json)
    return 0


def command_refresh(args: argparse.Namespace, adapters: Sequence[ProjectAdapter]) -> int:
    hosts = load_hosts(args.hosts_config)
    statuses = query_all_status(adapters, hosts, host_name=args.host)
    records = merged_records(adapters, include_central=False)
    if args.host:
        records = [record for record in records if record.get("host") == args.host]
    enriched_records = enrich_records_with_remote_status(records, adapters, hosts) if records else []
    known_run_ids = {str(record.get("run_id")) for record in records if record.get("run_id")}
    world = RunWorld.from_sources(enriched_records, statuses, known_run_ids=known_run_ids)
    current_records = world.current_records()
    anomalies = world.anomalies()
    payload = {
        "refreshed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "host:" + args.host if args.host else "all_hosts",
        "records": current_records,
        "anomalies": anomalies,
        "note": (
            "Server live processes are treated as current fact. "
            "History registry is not deleted or rewritten."
        ),
    }
    apply_refresh = not bool(getattr(args, "preview", False))
    if getattr(args, "apply", False):
        apply_refresh = True
    if apply_refresh:
        for adapter in adapters:
            current_path = adapter.local_root / ".remote_go" / "state" / "current.json"
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        payload["applied"] = True
    else:
        payload["applied"] = False
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(f"Refresh {'applied' if apply_refresh else 'preview'}: {len(current_records)} current row(s), {len(anomalies)} issue(s).")
    if apply_refresh:
        print("Wrote .remote_go/state/current.json")
    if current_records:
        print_registry(limit_current_records(current_records, args.limit), verbose=args.verbose)
    if anomalies:
        print()
        print_anomaly_report(anomalies, json_output=False)
    return 0


def command_adopt_live(args: argparse.Namespace, adapter: ProjectAdapter) -> int:
    hosts = load_hosts(args.hosts_config)
    host = find_host(hosts, args.host)
    status_payload = query_host_status(adapter, host)
    gpu, process = find_adoptable_live_process(adapter, status_payload, args.gpu, args.run_id)
    run_id = str(process["run_id"])
    if registry_has_run_id([adapter], run_id):
        raise ValueError(f"Run {run_id} already exists in the local registry; refusing to append a duplicate.")
    command_tokens = read_remote_process_argv(host, int(process["pid"]))
    metadata, status_payload, adoption_log = build_adopted_live_run(
        adapter=adapter,
        host=host,
        gpu=gpu,
        process=process,
        command_tokens=command_tokens,
        task_override=args.task,
        mode_override=args.mode,
        comment=args.comment,
        change_note=args.change_note,
    )
    if args.dry_run:
        print(json.dumps({
            "project": adapter.project_id,
            "host": host.name,
            "gpu": gpu["gpu"],
            "run_id": run_id,
            "pid": process.get("pid"),
            "release_dir": metadata["release_dir"],
            "remote_run_dir": remote_join(adapter.remote_project_root, "runs", run_id),
            "log_file": metadata["log_file"],
            "task": metadata["task"],
            "mode": metadata["mode"],
            "command": metadata["command"],
            "comment": metadata["comment"],
            "change_note": metadata["change_note"],
            "status": status_payload,
            "note": "dry-run only; no remote files or registry records were written.",
        }, indent=2, ensure_ascii=False))
        return 0

    write_adopted_remote_files(adapter, host, run_id, metadata, status_payload, adoption_log)
    append_registry(adapter.local_registry, metadata)
    print(f"Adopted live run {run_id}")
    print(f"Host/GPU/PID: {host.name} / {gpu['gpu']} / {process.get('pid')}")
    print(f"remote status: {remote_join(adapter.remote_project_root, 'runs', run_id, 'status.json')}")
    print(f"remote log: {metadata['log_file']}")
    return 0


def command_push(args: argparse.Namespace, adapter: ProjectAdapter) -> int:
    hosts = load_hosts(args.hosts_config)
    selected_hosts = [find_host(hosts, args.host)] if args.host else [hosts[0]]
    pushed = []
    for host in selected_hosts:
        target_dir = args.target_dir if args.target_dir is not None else adapter.push_target_dir
        remote_dir = adapter.remote_project_root if target_dir in {"", "."} else remote_join(adapter.remote_project_root, target_dir)
        if not is_under_remote_project_root(adapter, remote_dir):
            raise ValueError(f"Refusing to push outside remote project root: {remote_dir}")
        if args.dry_run:
            pushed.append(sync_project(adapter, host, remote_dir, dry_run=True))
            continue
        ensure_remote_dirs(adapter, host, [remote_dir])
        pushed.append(sync_project(adapter, host, remote_dir, dry_run=False))
    if args.dry_run:
        print(json.dumps({"push_targets": pushed, "note": "dry-run only; no ssh or rsync command was executed."}, indent=2))
        return 0
    for item in pushed:
        print(f"Ready on {item['target']}")
    return 0


def command_pull(args: argparse.Namespace, adapter: ProjectAdapter) -> int:
    hosts = load_hosts(args.hosts_config)
    selected_hosts = [find_host(hosts, args.host)] if args.host else list(hosts)
    pulled = []
    for host in selected_hosts:
        for spec in adapter.pull_specs:
            if args.kind != "all" and args.kind != spec.name:
                continue
            remote_dir = remote_join(adapter.remote_project_root, spec.remote_dir)
            if not is_under_remote_project_root(adapter, remote_dir):
                raise ValueError(f"Refusing to pull outside remote project root: {remote_dir}")
            if args.dry_run or remote_dir_exists(host, remote_dir):
                local_dir = adapter.local_root / spec.local_dir.format(host=host.name)
                pulled.append(pull_remote_tree(
                    host=host,
                    remote_dir=remote_dir,
                    local_dir=local_dir,
                    include_patterns=spec.include_patterns,
                    dry_run=args.dry_run,
                ))
    print(json.dumps({"pulled": pulled, "note": "dry-run only; no files were copied." if args.dry_run else "pull completed."}, indent=2))
    return 0


def record_adapter(records: Sequence[Dict[str, Any]], adapters: Sequence[ProjectAdapter], query: Optional[str]) -> Tuple[ProjectAdapter, Dict[str, Any]]:
    record = resolve_run_record(records, query)
    adapter_by_id = {adapter.project_id: adapter for adapter in adapters}
    adapter = adapter_by_id.get(record.get("project_id", ""))
    if adapter is None:
        raise ValueError(f"Run {record.get('run_id', '')} has no configured adapter.")
    return adapter, record


def command_log(args: argparse.Namespace, adapters: Sequence[ProjectAdapter]) -> int:
    records = merged_records(adapters, include_central=False)
    adapter, record = record_adapter(records, adapters, args.run_id)
    hosts = load_hosts(args.hosts_config)
    host = find_host(hosts, record["host"])
    proc = read_remote_log(host, record["log_file"], tail_lines=args.tail)
    if proc.stdout:
        print(strip_ansi(proc.stdout), end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return proc.returncode


def command_result(args: argparse.Namespace, adapters: Sequence[ProjectAdapter]) -> int:
    records = merged_records(adapters, include_central=False)
    adapter, record = record_adapter(records, adapters, args.run_id)
    hosts = load_hosts(args.hosts_config)
    host = find_host(hosts, record["host"])
    proc = grep_remote_result(host, record["log_file"], pattern=adapter.result_grep_pattern, lines=args.lines)
    if proc.stdout:
        print(strip_ansi(proc.stdout), end="" if proc.stdout.endswith("\n") else "\n")
    else:
        print(f"No metric-like lines found in {record['log_file']}.")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return proc.returncode


def command_run(args: argparse.Namespace, adapter: ProjectAdapter) -> int:
    task_name = args.task
    task_spec = adapter.resolve_task(task_name)
    task_name = task_spec.name
    hosts = load_hosts(args.hosts_config)
    command_tokens = adapter.build_command(task_name, args.framework_args)
    if not command_tokens:
        raise ValueError("go run requires a command after --, for example: ./go run -- python entrypoint.py")

    if args.dry_run:
        host = find_host(hosts, args.host) if args.host else hosts[0]
        selected_gpu = {"host": host.name, "gpu": args.gpu if args.gpu is not None else 0}
    else:
        statuses = query_all_status([adapter], hosts, host_name=args.host)
        try:
            selected_gpu = choose_idle_gpu(statuses, host_name=args.host, gpu_id=args.gpu)
        except RuntimeError as exc:
            print_status(statuses, json_output=False)
            print(f"\n{exc}")
            return 2
        host = find_host(hosts, selected_gpu["host"])

    mode_update = None
    if args.mode is not None:
        if args.dry_run:
            _, mode_update = build_mode_config_overlay(adapter, task_name, args.mode)
        else:
            mode_update = update_local_mode_config(adapter, task_name, args.mode)
    adapter_run_update: Dict[str, Any] = {}
    if adapter.prepare_run is not None:
        adapter_run_update = adapter.prepare_run(adapter, args, task_name, args.dry_run) or {}
    mode_label = args.mode or read_task_mode(adapter, task_name) or "config"
    required_artifacts = adapter.collect_required_artifacts(task_name, mode_label)

    run_id = make_run_id(adapter, task_name, mode_label, args.name)
    release_dir = remote_join(adapter.remote_project_root, "releases", run_id)
    remote_run_dir = remote_join(adapter.remote_project_root, "runs", run_id)
    remote_log_dir = remote_join(adapter.remote_project_root, "logs", "remote_go")
    remote_log_file = remote_join(remote_log_dir, f"{run_id}.log")
    remote_script_path = remote_join(remote_run_dir, "run.sh")
    remote_pane_script_path = remote_join(remote_run_dir, "pane.sh")
    remote_metadata_path = remote_join(remote_run_dir, "metadata.json")

    metadata = build_metadata(
        adapter=adapter,
        run_id=run_id,
        host=host,
        gpu_id=int(selected_gpu["gpu"]),
        release_dir=release_dir,
        log_file=remote_log_file,
        command_tokens=command_tokens,
        task=task_name,
        mode=mode_label,
        comment=args.comment,
        change_note=args.change_note,
    )
    metadata["mode_update"] = mode_update
    metadata["adapter_run_update"] = adapter_run_update
    metadata["required_artifacts"] = [relative_path for _, relative_path in required_artifacts]
    metadata["pane_script"] = remote_pane_script_path

    if args.dry_run:
        sync_preview = sync_project(adapter, host, release_dir, dry_run=True)
        print(json.dumps({
            "project": adapter.project_id,
            "selected": {"host": host.name, "gpu": selected_gpu["gpu"]},
            "release_dir": release_dir,
            "remote_run_dir": remote_run_dir,
            "remote_project_root": adapter.remote_project_root,
            "tmux": f"{adapter.tmux_session}:{adapter.tmux_window}",
            "tmux_pane": f"reuse_by_gpu:{adapter.display_name}_GPU_{selected_gpu['gpu']}",
            "command": metadata["command"],
            "mode_source": "local_config_then_sync" if args.mode is not None else "local_config",
            "mode_update": mode_update,
            "adapter_run_update": adapter_run_update,
            "required_artifacts": metadata["required_artifacts"],
            "registry": str(adapter.local_registry),
            "sync_preview": sync_preview,
            "comment": metadata["comment"],
            "change_note": metadata["change_note"],
            "note": "dry-run only; no local config change, ssh, rsync, registry write, or tmux command was executed.",
        }, indent=2))
        return 0

    ensure_remote_dirs(adapter, host, [release_dir, remote_run_dir, remote_log_dir])
    sync_project(adapter, host, release_dir, dry_run=False)
    metadata["uploaded_artifacts"] = sync_required_artifacts(adapter, host, release_dir, required_artifacts)

    run_script = make_remote_run_script(
        adapter=adapter,
        run_id=run_id,
        host=host,
        gpu_id=int(selected_gpu["gpu"]),
        release_dir=release_dir,
        remote_run_dir=remote_run_dir,
        log_file=remote_log_file,
        command_tokens=command_tokens,
        comment=args.comment,
        change_note=args.change_note,
    )
    pane_script = make_remote_pane_script(adapter, remote_script_path, int(selected_gpu["gpu"]))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        local_script = tmp_path / "run.sh"
        local_pane_script = tmp_path / "pane.sh"
        local_metadata = tmp_path / "metadata.json"
        local_script.write_text(run_script)
        local_pane_script.write_text(pane_script)
        local_metadata.write_text(json.dumps(metadata, indent=2))
        upload_file(host, local_script, remote_script_path)
        upload_file(host, local_pane_script, remote_pane_script_path)
        upload_file(host, local_metadata, remote_metadata_path)
    chmod_proc = run_ssh(
        host,
        f"chmod u+x {shlex.quote(remote_script_path)} {shlex.quote(remote_pane_script_path)}",
        capture=True,
        check=False,
    )
    if chmod_proc.returncode != 0:
        raise RuntimeError((chmod_proc.stderr or chmod_proc.stdout or "chmod failed").strip())
    launch_tmux_pane(adapter, host, remote_pane_script_path, int(selected_gpu["gpu"]), run_id)
    initial_status = wait_for_remote_run_status(adapter, host, run_id)
    metadata["initial_remote_status"] = initial_status
    append_registry(adapter.local_registry, metadata)
    if initial_status is None:
        raise RuntimeError(
            f"tmux command was sent, but run {run_id} did not create status.json within 10 seconds. "
            f"Inspect tmux session {adapter.tmux_session} on {host.name}."
        )
    print(f"Submitted {run_id}")
    print(f"Host/GPU: {host.name} / {selected_gpu['gpu']}")
    print(f"tmux: ssh {host.ssh} then tmux attach -t {adapter.tmux_session}")
    print(f"remote log: {remote_log_file}")
    return 0
