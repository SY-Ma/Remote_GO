from __future__ import annotations

import importlib.util
import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


ArtifactList = List[Tuple[Path, str]]
ArtifactCollector = Callable[["ProjectAdapter", str, Optional[str]], ArtifactList]
RunArgumentConfigurer = Callable[[Any], None]
RunPreparer = Callable[["ProjectAdapter", Any, str, bool], Dict[str, Any]]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    entrypoint: Sequence[str]
    mode_config_path: Optional[str] = None
    aliases: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class PullSpec:
    name: str
    remote_dir: str
    local_dir: str
    include_patterns: Sequence[str]


@dataclass
class ProjectAdapter:
    project_id: str
    project_label: str
    local_root: Path
    remote_project_root: str
    tasks: Dict[str, TaskSpec]
    rsync_exclude_patterns: Sequence[str]
    pull_specs: Sequence[PullSpec]
    result_grep_pattern: str
    tmux_session: str = "M"
    tmux_window: str = "M"
    conda_env: str = "pytorch"
    default_task: Optional[str] = None
    pane_title_prefix: Optional[str] = None
    legacy_env_prefix: Optional[str] = None
    collect_artifacts: Optional[ArtifactCollector] = None
    configure_run_parser: Optional[RunArgumentConfigurer] = None
    prepare_run: Optional[RunPreparer] = None
    local_registry_path: Optional[Path] = None
    push_target_dir: str = "workspace"

    @property
    def local_registry(self) -> Path:
        return self.local_registry_path or self.local_root / ".remote_go" / "state" / "registry.jsonl"

    @property
    def display_name(self) -> str:
        return self.pane_title_prefix or self.project_label

    def resolve_task(self, task_name: Optional[str]) -> TaskSpec:
        requested = task_name or self.default_task
        if requested is None:
            raise ValueError(f"Project {self.project_id} has no default task.")
        if requested in self.tasks:
            return self.tasks[requested]
        for spec in self.tasks.values():
            if requested in spec.aliases:
                return spec
        available = sorted(self.tasks)
        raise ValueError(f"Unsupported task {requested!r} for {self.project_id}. Available tasks: {available}.")

    def mode_config_path(self, task_name: str) -> Optional[Path]:
        spec = self.resolve_task(task_name)
        if spec.mode_config_path is None:
            return None
        return Path(spec.mode_config_path)

    def build_command(self, task_name: str, framework_args: Sequence[str]) -> List[str]:
        spec = self.resolve_task(task_name)
        args = list(framework_args)
        if args and args[0] == "--":
            args = args[1:]
        if not spec.entrypoint:
            return args
        return [*spec.entrypoint, *args]

    def collect_required_artifacts(self, task_name: str, mode: Optional[str]) -> ArtifactList:
        if self.collect_artifacts is None:
            return []
        return self.collect_artifacts(self, task_name, mode)

    def remote_join(self, *parts: str) -> str:
        clean_parts: List[str] = []
        for index, part in enumerate(parts):
            if index == 0:
                clean_parts.append(part.rstrip("/"))
            else:
                clean_parts.append(part.strip("/"))
        return posixpath.join(*clean_parts)


def load_adapter(adapter_path: Path) -> ProjectAdapter:
    path = adapter_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Adapter file does not exist: {path}")
    module = _load_module(path)
    if not hasattr(module, "build_adapter"):
        raise ValueError(f"Adapter {path} must define build_adapter().")
    adapter = module.build_adapter()
    if not isinstance(adapter, ProjectAdapter):
        raise TypeError(f"Adapter {path} returned {type(adapter).__name__}, expected ProjectAdapter.")
    validate_adapter(adapter)
    return adapter


def validate_adapter(adapter: ProjectAdapter) -> None:
    if not adapter.project_id:
        raise ValueError("Adapter project_id is required.")
    if not adapter.project_label:
        raise ValueError(f"Adapter {adapter.project_id} project_label is required.")
    adapter.local_root = adapter.local_root.expanduser().resolve()
    if not adapter.local_root.exists():
        raise FileNotFoundError(f"Local project root does not exist: {adapter.local_root}")
    if not adapter.local_root.is_dir():
        raise NotADirectoryError(f"Local project root is not a directory: {adapter.local_root}")
    if not adapter.remote_project_root.startswith("/"):
        raise ValueError(f"{adapter.project_id} remote_project_root must be absolute.")
    if not adapter.tasks:
        raise ValueError(f"{adapter.project_id} must define at least one task.")
    for name, spec in adapter.tasks.items():
        if name != spec.name:
            raise ValueError(f"Task dict key {name!r} does not match spec.name {spec.name!r}.")
        if not spec.entrypoint and name != "command":
            raise ValueError(f"Task {name!r} must define an entrypoint.")


def _load_module(path: Path) -> ModuleType:
    module_name = f"remote_go_adapter_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_relative_artifact(adapter: ProjectAdapter, artifacts: Dict[str, Path], path_value: Any) -> None:
    if path_value is None:
        return
    path_text = str(path_value)
    if path_text == "" or path_text.lower() == "null":
        return
    path = Path(path_text)
    if path.is_absolute():
        if path.exists():
            raise ValueError(
                f"Artifact path {path_text} is an absolute local path. Use a project-relative path so it can be synced."
            )
        return
    local_path = adapter.local_root / path
    if not local_path.exists():
        raise FileNotFoundError(
            f"Required artifact {path_text} does not exist locally for {adapter.project_id}. "
            "Use a project-relative path that exists, or an absolute path that already exists on the server."
        )
    if local_path.is_file():
        artifacts[str(path)] = local_path


def dedupe_artifacts(artifacts: Iterable[Tuple[Path, str]]) -> ArtifactList:
    deduped: Dict[str, Path] = {}
    for local_path, relative_path in artifacts:
        deduped[str(relative_path)] = local_path
    return [(local_path, relative_path) for relative_path, local_path in deduped.items()]
