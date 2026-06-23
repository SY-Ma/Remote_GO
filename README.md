# Remote_GO

English | [中文](README.zh-CN.md)

Remote_GO is a project-local command tool for SSH/tmux based remote experiment workflows. It helps you check remote GPU status, sync local files before launching experiments, track recent runs, and pull back selected logs or outputs.

It can also act as a stable helper layer for upper-layer AI assistants, so they can manage remote experiments through run ids, structured status, and predictable commands.

Remote_GO focuses on the common local-to-remote experiment workflow.

![Remote_GO overview](docs/assets/remote-go-overview.svg)

PDF version: [remote-go-overview.pdf](docs/assets/remote-go-overview.pdf)

## When To Use It

| Scenario | What Remote_GO does |
| --- | --- |
| You edit code locally in an IDE and run it on one or more remote GPU servers | Run one local command to upload the project and start the code on a selected server/GPU |
| Your experiments are spread across several remote servers | Show server and GPU status in one local view |
| You often run multiple experiments for the same project | Show recent runs, their status, host, GPU, command, and log location |
| You need to check a running or finished experiment | Tail the remote log from your local terminal |
| You want selected logs or results back on your laptop | Pull only the configured logs or output files back into the local project |
| You want an upper-layer AI assistant to help with remote experiments | Provide a stable local tool the AI can read and call instead of parsing scattered server commands |

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

Recommended for most users: download Remote_GO from GitHub with `Code -> Download ZIP`, unzip it, rename the folder to `Remote_GO`, then place it inside your project root.

```text
your_project/
  Remote_GO/
  train_or_entrypoint.py
```

Install the local Python dependency and initialize the project:

```bash
cd /path/to/your_project
python -m pip install -r Remote_GO/requirements.txt
./Remote_GO/go init
```

After that, use the generated project-local command:

```bash
./go status
```

Engineering option from your project root:

```bash
git clone https://github.com/SY-Ma/Remote_GO.git Remote_GO
python -m pip install -r Remote_GO/requirements.txt
./Remote_GO/go init
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
  id: my_project          # required; short stable id used inside run_id values
  label: My Project       # optional; human-readable project name shown in tables

remote:
  root: /home/my_user/projects/my_project  # required; absolute project directory on each remote host
  env:
    type: conda           # required; currently only conda is supported
    name: pytorch         # required; remote conda environment name

tmux:
  session: M              # optional; tmux session name, default is M
  window: M               # optional; tmux window name, default is M

hosts:
  # Hosts are tried from top to bottom when --host is not provided.
  - name: gpu1            # required; short host name used by ./go --host
    ssh: my_user@gpu1     # required; SSH target accepted by your local ssh command
  - name: gpu2            # optional; add more hosts if you use more than one server
    ssh: my_user@gpu2     # required for each host

sync:
  push_exclude_file: .remote_go/push.exclude  # optional; upload ignore rules
  pull_rules_file: .remote_go/pull.yaml       # optional; download allow-list rules
  push_target: workspace                      # optional; go push uploads to remote.root/workspace/
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
| `./go run -- python entrypoint.py` | Sync the current local project, then launch a remote tmux run |
| `./go runs [--limit 30]` | Show recent run records. Default limit is 12 |
| `./go log <run_id>` | Tail one remote run log |
| `./go kill <run_id>` | Stop one of your own runs |
| `./go push [--host gpu1]` | Manually sync project files to the configured remote workspace |
| `./go pull` | Manually pull configured logs, outputs, and model files back locally |
| `./go refresh` | Rebuild the current run view from live server facts |

Launch examples:

```bash
./go run --dry-run -- python entrypoint.py --config configs/train.yaml
./go run --host gpu1 --gpu 0 --name baseline -- python entrypoint.py --epochs 100
```

`./go run` runs the upload step first, then starts the command in the configured remote tmux session. You usually do not need to run `./go push` before each experiment; use `./go push` when you want to update the remote workspace without launching a run.

Use `./go kill <run_id> --dry-run` if you want to preview which remote process would be stopped.

Watch directly on the server:

```bash
ssh my_user@gpu1
tmux attach -t M
```

Use the SSH target from `hosts[].ssh` and the session name from `tmux.session` in `.remote_go/config.yaml`. Remote_GO also prints this tmux attach hint after `./go run`.

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

Remote_GO can provide a stable local interface for an upper-layer AI assistant. Useful machine-readable commands include:

- `./go status --json`: read current host/GPU/process state.
- `./go runs --json`: read recent runs with stable `run_id`, host, GPU, command, log path, and status.
- `./go refresh --json`: rebuild the current run view from live server facts.
- `./go run --dry-run -- ...` and `./go kill <run_id> --dry-run`: preview remote actions.

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

- `go run` syncs the current local project into `remote.root/releases/<run_id>/` before launching.
- `go push` manually syncs the current local project to `remote.root/workspace/` by default.
- `go pull` uses allow-list rules from `.remote_go/pull.yaml`.
- Remote paths are rejected if they escape `remote.root`.
- History is append-only in `.remote_go/state/registry.jsonl`.
- `go refresh` writes `.remote_go/state/current.json` without rewriting history. Use `--preview` to avoid writing.
- `go kill` only signals current-user processes that Remote_GO can prove belong to the current project.

## License

Remote_GO is released under the MIT License. See [LICENSE](LICENSE).
