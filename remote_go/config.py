from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit("PyYAML is required. Install it in the local environment first.") from exc

from .core.adapter import ProjectAdapter, PullSpec, TaskSpec, validate_adapter


CONFIG_DIR = ".remote_go"
CONFIG_FILE = "config.yaml"
PUSH_EXCLUDE_FILE = "push.exclude"
PULL_RULES_FILE = "pull.yaml"
STATE_DIR = "state"
REGISTRY_FILE = "registry.jsonl"
CURRENT_FILE = "current.json"


DEFAULT_PUSH_EXCLUDES = [
    ".git/",
    ".DS_Store",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".ipynb_checkpoints/",
    ".env",
    ".env.*",
    ".venv/",
    "venv/",
    "env/",
    ".remote_go/state/",
    "remote_runs/",
    "logs/",
    "logs_server/",
    "outputs/",
    "wandb/",
    "*.log",
    "*.pt",
    "*.pth",
    "*.ckpt",
]


DEFAULT_PULL_RULES = {
    "kinds": {
        "logs": {
            "remote": "logs/",
            "local": "logs_server/{host}/",
            "include": ["*/", "*.log", "*.txt", "*.json"],
        },
        "outputs": {
            "remote": "outputs/",
            "local": "outputs_server/{host}/",
            "include": ["*/", "*.csv", "*.json", "*.png", "*.jpg", "*.jpeg", "*.txt"],
        },
    }
}


def sanitize_project_id(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "_")
    safe = re.sub(r"[^a-z0-9_-]+", "_", lowered).strip("_-")
    return safe or "project"


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / CONFIG_DIR / CONFIG_FILE).exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {CONFIG_DIR}/{CONFIG_FILE}. Run ./go init from your project root first."
    )


def project_config_path(project_root: Path) -> Path:
    return project_root / CONFIG_DIR / CONFIG_FILE


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def default_config(project_root: Path) -> Dict[str, Any]:
    project_id = sanitize_project_id(project_root.name)
    return {
        "project": {
            "id": project_id,
            "label": project_root.name,
        },
        "remote": {
            "root": f"/home/YOUR_SSH_USER/projects/{project_root.name}",
            "env": {
                "type": "conda",
                "name": "pytorch",
            },
        },
        "tmux": {
            "session": "M",
            "window": "M",
        },
        "hosts": [
            {
                "name": "gpu1",
                "ssh": "YOUR_SSH_USER@gpu1",
                "idle_mem_threshold_mib": 100,
                "idle_util_threshold_percent": 8,
                "ssh_connect_timeout": 8,
            },
            {
                "name": "gpu2",
                "ssh": "YOUR_SSH_USER@gpu2",
                "idle_mem_threshold_mib": 100,
                "idle_util_threshold_percent": 8,
                "ssh_connect_timeout": 8,
            },
        ],
        "sync": {
            "push_exclude_file": f"{CONFIG_DIR}/{PUSH_EXCLUDE_FILE}",
            "pull_rules_file": f"{CONFIG_DIR}/{PULL_RULES_FILE}",
            "push_target": "workspace",
        },
    }


def default_config_text(project_root: Path) -> str:
    project_id = sanitize_project_id(project_root.name)
    return f"""# Remote_GO project configuration.
# Edit the fields marked "CHANGE ME" before launching real remote runs.
# Local project root is detected automatically as the directory containing this .remote_go folder.

project:
  # CHANGE ME only if you want a shorter stable id in run_id values.
  # Use lowercase letters, numbers, underscore, or dash.
  id: {project_id}

  # CHANGE ME to the human-readable project name shown in status/runs tables.
  label: {project_root.name}

remote:
  # CHANGE ME to an absolute directory on each remote host.
  # Remote_GO creates releases/<run_id>/, runs/<run_id>/, logs/, and workspace/ under this root.
  root: /home/YOUR_SSH_USER/projects/{project_root.name}

  env:
    # Currently supported: conda.
    type: conda

    # CHANGE ME to the remote conda environment name.
    # This is the remote environment, not your local Python environment.
    name: pytorch

tmux:
  # CHANGE ME only if you already use a different tmux session/window convention.
  session: M
  window: M

hosts:
  # CHANGE ME. Hosts are tried from top to bottom when --host is not provided.
  # name is the short label used by ./go commands; ssh is the actual SSH target.
  - name: gpu1
    ssh: YOUR_SSH_USER@gpu1
    idle_mem_threshold_mib: 100
    idle_util_threshold_percent: 8
    ssh_connect_timeout: 8

  # Optional second host. Delete this block if you only have one server.
  - name: gpu2
    ssh: YOUR_SSH_USER@gpu2
    idle_mem_threshold_mib: 100
    idle_util_threshold_percent: 8
    ssh_connect_timeout: 8

sync:
  # Files ignored when pushing code to remote releases/workspace.
  push_exclude_file: .remote_go/push.exclude

  # Allow-list rules for pulling logs/results back from remote hosts.
  pull_rules_file: .remote_go/pull.yaml

  # go push syncs current local project files to remote.root/<push_target>/.
  # Keep workspace unless you want a different remote-side copy folder.
  push_target: workspace
"""


def default_pull_rules_text() -> str:
    return """# Remote_GO pull rules.
# remote paths are relative to remote.root on the selected host.
# local paths are relative to your local project root.
# {host} is replaced with the configured host name, such as gpu1.

kinds:
  logs:
    remote: logs/
    local: logs_server/{host}/
    include:
      - "*/"
      - "*.log"
      - "*.txt"
      - "*.json"

  outputs:
    remote: outputs/
    local: outputs_server/{host}/
    include:
      - "*/"
      - "*.csv"
      - "*.json"
      - "*.png"
      - "*.jpg"
      - "*.jpeg"
      - "*.txt"
"""


def init_project(project_root: Path, force: bool = False) -> List[Path]:
    root = project_root.resolve()
    config_dir = root / CONFIG_DIR
    state_dir = config_dir / STATE_DIR
    created: List[Path] = []
    config_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / CONFIG_FILE
    if force or not config_path.exists():
        config_path.write_text(default_config_text(root))
        created.append(config_path)

    push_path = config_dir / PUSH_EXCLUDE_FILE
    if force or not push_path.exists():
        push_path.write_text("\n".join(DEFAULT_PUSH_EXCLUDES) + "\n")
        created.append(push_path)

    pull_path = config_dir / PULL_RULES_FILE
    if force or not pull_path.exists():
        pull_path.write_text(default_pull_rules_text())
        created.append(pull_path)

    gitignore_block_path = config_dir / "gitignore.block"
    gitignore_block = "\n".join([
        "# Remote_GO begin",
        ".remote_go/state/",
        "remote_runs/",
        "logs_server/",
        "outputs_server/",
        "# Remote_GO end",
        "",
    ])
    if force or not gitignore_block_path.exists():
        gitignore_block_path.write_text(gitignore_block)
        created.append(gitignore_block_path)

    gitignore_path = root / ".gitignore"
    existing_gitignore = gitignore_path.read_text() if gitignore_path.exists() else ""
    if "# Remote_GO begin" not in existing_gitignore:
        prefix = "" if not existing_gitignore or existing_gitignore.endswith("\n") else "\n"
        gitignore_path.write_text(existing_gitignore + prefix + gitignore_block)
        created.append(gitignore_path)

    registry_path = state_dir / REGISTRY_FILE
    if force or not registry_path.exists():
        registry_path.write_text("")
        created.append(registry_path)

    go_path = root / "go"
    tool_root = Path(__file__).resolve().parents[1]
    wrapper = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TOOL_ROOT={str(tool_root)!r}

if python -c "import remote_go" >/dev/null 2>&1; then
  python -m remote_go.cli "$@"
elif [ -d "$ROOT_DIR/remote_go" ]; then
  PYTHONPATH="$ROOT_DIR${{PYTHONPATH:+:$PYTHONPATH}}" python -m remote_go.cli "$@"
elif [ -d "$TOOL_ROOT/remote_go" ]; then
  PYTHONPATH="$TOOL_ROOT${{PYTHONPATH:+:$PYTHONPATH}}" python -m remote_go.cli "$@"
else
  echo "Remote_GO is not installed and no embedded remote_go package was found." >&2
  exit 1
fi
"""
    if force or not go_path.exists():
        go_path.write_text(wrapper)
        os.chmod(go_path, 0o755)
        created.append(go_path)

    return created


def load_pull_specs(project_root: Path, config: Dict[str, Any]) -> Tuple[PullSpec, ...]:
    sync = config.get("sync", {}) or {}
    pull_rules_value = sync.get("pull_rules_file", f"{CONFIG_DIR}/{PULL_RULES_FILE}")
    pull_rules_path = (project_root / pull_rules_value).resolve()
    rules = read_yaml(pull_rules_path) or DEFAULT_PULL_RULES
    specs: List[PullSpec] = []
    for name, spec in (rules.get("kinds", {}) or {}).items():
        specs.append(PullSpec(
            name=str(name),
            remote_dir=str(spec.get("remote", "")),
            local_dir=str(spec.get("local", "")),
            include_patterns=tuple(str(item) for item in spec.get("include", [])),
        ))
    return tuple(specs)


def load_push_excludes(project_root: Path, config: Dict[str, Any]) -> Tuple[str, ...]:
    sync = config.get("sync", {}) or {}
    exclude_value = sync.get("push_exclude_file", f"{CONFIG_DIR}/{PUSH_EXCLUDE_FILE}")
    exclude_path = (project_root / exclude_value).resolve()
    if not exclude_path.exists():
        return tuple(DEFAULT_PUSH_EXCLUDES)
    patterns = [
        line.strip()
        for line in exclude_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return tuple(patterns)


def load_project(project_root: Path | None = None) -> Tuple[ProjectAdapter, Path]:
    root = (project_root.resolve() if project_root is not None else find_project_root())
    config_path = project_config_path(root)
    config = read_yaml(config_path)
    project = config.get("project", {}) or {}
    remote = config.get("remote", {}) or {}
    env = remote.get("env", {}) or {}
    tmux = config.get("tmux", {}) or {}
    sync = config.get("sync", {}) or {}

    adapter = ProjectAdapter(
        project_id=sanitize_project_id(str(project.get("id") or root.name)),
        project_label=str(project.get("label") or root.name),
        local_root=root,
        remote_project_root=str(remote.get("root") or ""),
        tasks={"command": TaskSpec(name="command", entrypoint=())},
        rsync_exclude_patterns=load_push_excludes(root, config),
        pull_specs=load_pull_specs(root, config),
        result_grep_pattern=str(config.get("result_grep_pattern", "Epoch|VAL:|TEST:|Final|Result|MAE|MSE")),
        tmux_session=str(tmux.get("session", "M")),
        tmux_window=str(tmux.get("window", "M")),
        conda_env=str(env.get("name", "pytorch")),
        default_task="command",
        pane_title_prefix=str(project.get("label") or root.name),
        local_registry_path=root / CONFIG_DIR / STATE_DIR / REGISTRY_FILE,
        push_target_dir=str(sync.get("push_target", "workspace")),
    )
    validate_adapter(adapter)
    return adapter, config_path
