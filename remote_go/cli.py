from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import init_project, load_project
from .core.console import (
    command_kill,
    command_log,
    command_pull,
    command_push,
    command_refresh,
    command_run,
    command_runs,
    command_status,
)


def add_project_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing .remote_go/config.yaml. Defaults to the nearest parent with that file.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="go",
        description="Remote_GO: lightweight SSH/tmux GPU experiment helper for one project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create .remote_go config, sync rules, state files, and ./go.")
    init_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing Remote_GO files.")
    init_parser.set_defaults(func="init")

    status_parser = subparsers.add_parser("status", help="Show GPU status across configured hosts.")
    add_project_config(status_parser)
    status_parser.add_argument("--host", type=str, default=None)
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func="status")

    run_parser = subparsers.add_parser("run", help="Sync the project and start a remote tmux run.")
    add_project_config(run_parser)
    run_parser.add_argument("--host", type=str, default=None)
    run_parser.add_argument("--gpu", type=int, default=None)
    run_parser.add_argument("--name", type=str, default=None)
    run_parser.add_argument("-c", "--comment", type=str, default=None)
    run_parser.add_argument("--change-note", type=str, default=None)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("framework_args", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func="run", task="command", mode=None)

    runs_parser = subparsers.add_parser("runs", help="Show run records. Default limit: 12.")
    add_project_config(runs_parser)
    runs_parser.add_argument("--limit", type=int, default=12)
    runs_parser.add_argument("--all", action="store_true", help="Show all historical records.")
    runs_parser.add_argument("--local-only", action="store_true")
    runs_parser.add_argument("--verbose", action="store_true")
    runs_parser.add_argument("--history", action="store_true")
    runs_parser.set_defaults(func="runs")

    log_parser = subparsers.add_parser("log", help="Show the remote log tail for one run.")
    add_project_config(log_parser)
    log_parser.add_argument("run_id", nargs="?", default=None)
    log_parser.add_argument("--tail", type=int, default=120)
    log_parser.set_defaults(func="log")

    kill_parser = subparsers.add_parser("kill", help="Signal one of your own Remote_GO runs by visible run id key.")
    add_project_config(kill_parser)
    kill_parser.add_argument("key", help="Run id, prefix, suffix, or unique substring shown by go runs/status.")
    kill_parser.add_argument("--host", type=str, default=None)
    kill_parser.add_argument("--gpu", type=int, default=None)
    kill_parser.add_argument("--signal", default="TERM")
    kill_parser.add_argument("--dry-run", action="store_true")
    kill_parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    kill_parser.add_argument("--all", action="store_true", help="Signal all live candidates for the same run id.")
    kill_parser.set_defaults(func="kill")

    push_parser = subparsers.add_parser("push", help="Push project files to the configured remote server.")
    add_project_config(push_parser)
    push_parser.add_argument("--host", type=str, default=None)
    push_parser.add_argument("--target-dir", type=str, default=None)
    push_parser.add_argument("--dry-run", action="store_true")
    push_parser.set_defaults(func="push")

    pull_parser = subparsers.add_parser("pull", help="Pull configured remote files back to the local project.")
    add_project_config(pull_parser)
    pull_parser.add_argument("--host", type=str, default=None)
    pull_parser.add_argument("--kind", type=str, default="all")
    pull_parser.add_argument("--dry-run", action="store_true")
    pull_parser.set_defaults(func="pull")

    refresh_parser = subparsers.add_parser("refresh", help="Rebuild the current run view from live server facts.")
    add_project_config(refresh_parser)
    refresh_parser.add_argument("--host", type=str, default=None)
    refresh_parser.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)
    refresh_parser.add_argument("--preview", action="store_true", help="Show the rebuilt current view without writing current.json.")
    refresh_parser.add_argument("--json", action="store_true")
    refresh_parser.add_argument("--limit", type=int, default=12)
    refresh_parser.add_argument("--verbose", action="store_true")
    refresh_parser.set_defaults(func="refresh")

    return parser


def load_context(args: argparse.Namespace):
    adapter, config_path = load_project(args.project_root)
    args.hosts_config = config_path
    return adapter


def dispatch(args: argparse.Namespace) -> int:
    if args.func == "init":
        created = init_project(args.project_root, force=args.force)
        print("Remote_GO initialized.")
        for path in created:
            print(f"  {path}")
        return 0

    adapter = load_context(args)
    if args.func == "status":
        return command_status(args, [adapter])
    if args.func == "run":
        return command_run(args, adapter)
    if args.func == "runs":
        args.all_records = bool(args.all)
        return command_runs(args, [adapter])
    if args.func == "log":
        return command_log(args, [adapter])
    if args.func == "kill":
        return command_kill(args, adapter)
    if args.func == "push":
        return command_push(args, adapter)
    if args.func == "pull":
        pull_choices = {"all", *{spec.name for spec in adapter.pull_specs}}
        if args.kind not in pull_choices:
            raise ValueError(f"Unknown pull kind {args.kind!r}. Available: {sorted(pull_choices)}.")
        return command_pull(args, adapter)
    if args.func == "refresh":
        return command_refresh(args, [adapter])
    raise ValueError(f"Unsupported command {args.func}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return dispatch(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Remote_GO error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
