# Remote_GO

English | [中文](README.zh-CN.md)

Remote_GO is a project-local command tool for SSH/tmux based remote experiment workflows. It helps you keep a lightweight view of remote GPU status, launch your own experiments, track recent runs, and pull back selected logs or outputs.

It also gives AI assistants and scripts a clean management layer. Instead of asking an AI to infer server state from scattered shell commands, Remote_GO keeps stable run ids, project-local records, structured status, and predictable commands that are easier to read, verify, and call safely.

It is intentionally small. Remote_GO is not a scheduler, cloud platform, queue system, or full experiment tracker.

## When To Use It

| Scenario | What Remote_GO does |
| --- | --- |
| You edit code locally in an IDE and run it on one or more remote GPU servers | Run one local command to upload the project and start the code on a selected server/GPU |
| Your experiments are spread across several remote servers | Show server and GPU status in one local view |
| You often run multiple experiments for the same project | Show recent runs, their status, host, GPU, command, and log location |
| You need to check a running or finished experiment | Tail the remote log from your local terminal |
| You want selected logs or results back on your laptop | Pull only the configured logs or output files back into the local project |
| You want an AI assistant to help manage remote experiments | Give the AI stable run ids, readable state, and JSON command output instead of scattered server commands |

## Requirements

`requirements.txt` covers local Python package dependencies only. SSH, rsync, tmux, and GPU tools are system dependencies.

Local machine:

- Python 3.10+
- SSH access to the remote hosts
- `rsync`
- Python dependencies from `requirements.txt`

Remote hosts:

- Linux
- `bash`, `python3`, `rsync`, `tmux`, `flock`
- NVIDIA driver tools with `nvidia-smi`
- A conda environment for running your experiments

## Install

Clone Remote_GO and install its local Python dependency:

```bash
git clone https://github.com/SY-Ma/Remote_GO.git
python -m pip install -r Remote_GO/requirements.txt
```

Initialize your own project from its project root:

```bash
cd /path/to/your_project
/path/to/Remote_GO/go init
```

After that, use the generated project-local command:

```bash
./go status
```

## Configure A Project

`init` creates these files in your project:

```text
.remote_go/
  config.yaml        # hosts, remote root, conda env, tmux names
  push.exclude       # files ignored by go run/go push
  pull.yaml          # allow-list rules for go pull
  gitignore.block    # generated .gitignore block for Remote_GO state
  state/
    registry.jsonl   # append-only run history
go                   # project-local command wrapper
```

Edit `.remote_go/config.yaml` first:

```yaml
project:
  id: my_project
  label: My Project

remote:
  root: /home/my_user/projects/my_project
  env:
    type: conda
    name: pytorch

tmux:
  session: M
  window: M

hosts:
  # Hosts are tried from top to bottom when --host is not provided.
  - name: gpu1
    ssh: my_user@gpu1
  - name: gpu2
    ssh: my_user@gpu2

sync:
  push_exclude_file: .remote_go/push.exclude
  pull_rules_file: .remote_go/pull.yaml
  push_target: workspace  # go push uploads to remote.root/workspace/
```

Fields you normally fill:

- Local project root is detected automatically. It is the directory that contains `.remote_go/config.yaml`.
- `remote.root`: absolute directory on each remote host.
- `remote.env.name`: remote conda environment name, not your local environment.
- `hosts[].name`: short name used by `./go --host`.
- `hosts[].ssh`: SSH target accepted by your local `ssh` command.
- `sync.push_target`: remote subfolder used by `./go push`. Keep `workspace` unless you want a different remote copy folder.
- `.remote_go/push.exclude`: files skipped during upload.
- `.remote_go/pull.yaml`: allowed remote folders and file patterns for download.

## Commands

| Command | Purpose |
| --- | --- |
| `./go init` | Create Remote_GO files in a project |
| `./go status [--host gpu1]` | Show configured remote GPU status |
| `./go run -- python entrypoint.py` | Push code and launch a remote tmux run |
| `./go runs [--limit 30]` | Show recent run records. Default limit is 12 |
| `./go log <run_id>` | Tail one remote run log |
| `./go kill <run_id>` | Stop one of your own runs |
| `./go push [--host gpu1]` | Push project files to the configured remote workspace |
| `./go pull` | Pull configured logs, outputs, and model files back locally |
| `./go refresh` | Rebuild the current run view from live server facts |

Launch examples:

```bash
./go run --dry-run -- python entrypoint.py --config configs/train.yaml
./go run --host gpu1 --gpu 0 --name baseline -- python entrypoint.py --epochs 100
```

Use `./go kill <run_id> --dry-run` if you want to preview which remote process would be stopped.

Run list examples:

```bash
./go runs
./go runs --limit 30
./go runs --all
./go runs --json
```

Pull examples:

```bash
./go pull
./go pull --host gpu1
```

Edit `.remote_go/pull.yaml` to choose which logs, outputs, or model files are copied.

## AI-Friendly Use

Remote_GO is designed to stay easy for people first, while still being predictable for an AI layer above it. The important surfaces are:

- `./go status --json`: read current host/GPU/process state.
- `./go runs --json`: read recent runs with stable `run_id`, host, GPU, command, log path, and status.
- `./go refresh --json`: rebuild the current run view from live server facts.
- `./go run --dry-run -- ...` and `./go kill <run_id> --dry-run`: preview remote actions before executing them.

This makes it easier for an AI assistant to manage experiments accurately because it can refer to one project-local interface instead of guessing from shell history, tmux panes, process names, or ad hoc log files.

## Build And Validate

For local development:

```bash
python -m unittest discover -s tests -v
```

To build a distributable package:

```bash
python -m pip install build
python -m build
```

## Safety Notes

- `go run` syncs into `remote.root/releases/<run_id>/`.
- `go push` defaults to `remote.root/workspace/`.
- `go pull` uses allow-list rules from `.remote_go/pull.yaml`.
- Remote paths are rejected if they escape `remote.root`.
- History is append-only in `.remote_go/state/registry.jsonl`.
- `go refresh` writes `.remote_go/state/current.json` without rewriting history. Use `--preview` to avoid writing.
- `go kill` only signals current-user processes that Remote_GO can prove belong to the current project.

## License

Remote_GO is released under the MIT License. See [LICENSE](LICENSE).
