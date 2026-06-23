# Remote_GO

English | [中文](README.zh-CN.md)

Remote_GO is a project-local command tool for SSH/tmux based remote experiment workflows. It helps you keep a lightweight view of remote GPU status, launch your own experiments, track recent runs, and pull back selected logs or outputs.

It is intentionally small. Remote_GO is not a scheduler, cloud platform, queue system, or full experiment tracker.

## When To Use It

| Scenario | Remote_GO helps with | Out of scope |
| --- | --- | --- |
| One local codebase, fixed lab servers | Keep host priority, SSH targets, remote root, and conda environment in one project config | Managing dynamic cloud machines |
| Students or researchers sharing a few GPU boxes | See which GPUs are idle, busy by you, busy by others, or occupied by untracked local processes | Fair-share scheduling or quotas |
| Daily experiment launches | Push the project, start a tmux run, assign a stable `RUN_ID`, and write a local run record | Replacing Slurm, Kubernetes, or Weights & Biases |
| Checking recent work | Show the latest runs, tail logs, and rebuild the current view from live server facts | Long-term metrics storage |
| Moving results home | Pull only allowed logs or outputs through editable allow-list rules | Unrestricted remote directory mirroring |

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

Clone the repository:

```bash
git clone https://github.com/<your-name>/Remote_GO.git
cd Remote_GO
```

Install local Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Use either source mode:

```bash
./go init --project-root /path/to/your_project
```

or install the package in editable mode:

```bash
python -m pip install -e .
remote-go init --project-root /path/to/your_project
```

After `init`, run commands from your own project with the generated wrapper:

```bash
cd /path/to/your_project
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
  push_target: workspace
```

Fields you normally fill:

- `remote.root`: absolute directory on each remote host.
- `remote.env.name`: remote conda environment name, not your local environment.
- `hosts[].name`: short name used by `./go --host`.
- `hosts[].ssh`: SSH target accepted by your local `ssh` command.
- `.remote_go/push.exclude`: files skipped during upload.
- `.remote_go/pull.yaml`: allowed remote folders and file patterns for download.

## Commands

| Command | Purpose |
| --- | --- |
| `./go init` | Create Remote_GO files in a project |
| `./go status [--host gpu1]` | Show configured remote GPU status |
| `./go run -- python train.py` | Push code and launch a remote tmux run |
| `./go runs [--limit 30]` | Show recent run records. Default limit is 12 |
| `./go log <run_id>` | Tail one remote run log |
| `./go kill <run_id> --dry-run` | Preview stopping one of your own tracked runs |
| `./go kill <run_id> --yes` | Stop one confirmed Remote_GO run |
| `./go push [--host gpu1]` | Push project files to `remote.root/workspace/` |
| `./go pull --kind logs` | Pull allowed logs or outputs back locally |
| `./go refresh --apply` | Rebuild the current run view from live server facts |

Launch examples:

```bash
./go run --dry-run -- python train.py --config configs/train.yaml
./go run --host gpu1 --gpu 0 --name baseline -- python train.py --epochs 100
```

Run list examples:

```bash
./go runs
./go runs --limit 30
./go runs --all
```

Pull examples:

```bash
./go pull --kind logs
./go pull --kind outputs --host gpu1
```

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
- `go refresh --apply` writes `.remote_go/state/current.json` without rewriting history.
- `go kill` only signals current-user processes that Remote_GO can prove belong to the current project.

## License

Remote_GO is released under the MIT License. See [LICENSE](LICENSE).
