# Remote_GO

[English](README.md) | 中文

Remote_GO 是一个项目本地化的 SSH/tmux 远程实验命令工具。它用于读取远程 GPU 状态、启动自己的实验、跟踪最近运行记录，并按规则把日志或结果拉回本地。

它也可以作为 AI 助手或脚本管理远程实验的干净接口。相比让 AI 从零散的 shell 命令、tmux 窗口和进程名里猜测服务器状态，Remote_GO 提供稳定的 run id、项目本地记录、结构化状态和可预期的命令，更容易被 AI 准确读取、验证和调用。

它刻意保持轻量。Remote_GO 不是调度系统、云平台、队列系统，也不是完整实验管理平台。

## 适用场景

| 场景 | Remote_GO 能做什么 |
| --- | --- |
| 你在本地 IDE 修改代码，但需要在一台或多台远程 GPU 服务器上运行 | 用一条本地命令上传项目，并在指定服务器/GPU 上执行代码 |
| 实验分布在多台远程服务器上 | 用本地指令集中查看服务器和 GPU 设备状态 |
| 同一个项目经常同时跑多个实验 | 查看最近执行过的 run、当前状态、所在服务器、GPU、命令和日志位置 |
| 需要检查正在运行或已经结束的实验 | 直接在本地终端查看远程日志 |
| 想把日志或结果拉回本地 | 按配置规则只拉回需要的日志或输出文件 |
| 想让 AI 助手协助管理远程实验 | 给 AI 提供稳定的 run id、可读状态和 JSON 输出，而不是让它直接解析零散服务器命令 |

## 环境要求

`requirements.txt` 只负责本地 Python 包依赖。SSH、rsync、tmux、GPU 工具属于系统依赖，不能只靠 `requirements.txt` 安装。

本地机器：

- Python 3.10+
- 能通过 SSH 访问远程服务器
- `rsync`
- `requirements.txt` 中的 Python 依赖

远程服务器：

- Linux
- `bash`、`python3`、`rsync`、`tmux`、`flock`
- NVIDIA 驱动工具，能执行 `nvidia-smi`
- 用于运行实验的 conda 环境

## 下载和安装

克隆 Remote_GO，并安装本地 Python 依赖：

```bash
git clone https://github.com/SY-Ma/Remote_GO.git
python -m pip install -r Remote_GO/requirements.txt
```

进入你自己的项目根目录并初始化：

```bash
cd /path/to/your_project
/path/to/Remote_GO/go init
```

之后在你的项目里使用生成的本地命令：

```bash
./go status
```

## 配置项目

`init` 会在你的项目中生成：

```text
.remote_go/
  config.yaml        # hosts、远程根目录、conda 环境、tmux 名称
  push.exclude       # go run/go push 上传时忽略的文件
  pull.yaml          # go pull 拉取时使用的白名单规则
  gitignore.block    # Remote_GO 状态文件对应的 .gitignore 片段
  state/
    registry.jsonl   # 追加式运行历史
go                   # 项目本地命令包装脚本
```

首先修改 `.remote_go/config.yaml`：

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
  # 不指定 --host 时，会按从上到下的顺序尝试主机。
  - name: gpu1
    ssh: my_user@gpu1
  - name: gpu2
    ssh: my_user@gpu2

sync:
  push_exclude_file: .remote_go/push.exclude
  pull_rules_file: .remote_go/pull.yaml
  push_target: workspace  # go push 会上传到 remote.root/workspace/
```

通常需要填写这些字段：

- 本地项目根目录会自动识别，就是包含 `.remote_go/config.yaml` 的目录。
- `remote.root`：每台远程服务器上的绝对路径。
- `remote.env.name`：远程 conda 环境名，不是本地 Python 环境。
- `hosts[].name`：`./go --host` 使用的短名称。
- `hosts[].ssh`：本地 `ssh` 命令可直接识别的 SSH 目标。
- `sync.push_target`：`./go push` 使用的远程子目录。一般保持 `workspace` 即可。
- `.remote_go/push.exclude`：上传时跳过的文件。
- `.remote_go/pull.yaml`：允许拉回的远程目录和文件模式。

## 命令

| 命令 | 作用 |
| --- | --- |
| `./go init` | 在项目中创建 Remote_GO 文件 |
| `./go status [--host gpu1]` | 查看配置的远程 GPU 状态 |
| `./go run -- python 程序入口.py` | 推送代码并启动远程 tmux 实验 |
| `./go runs [--limit 30]` | 查看最近运行记录，默认显示 12 条 |
| `./go log <run_id>` | 查看某次远程实验的日志尾部 |
| `./go kill <run_id>` | 停止自己的某个 run |
| `./go push [--host gpu1]` | 推送项目文件到配置好的远程工作目录 |
| `./go pull` | 把配置好的日志、输出和模型文件拉回本地 |
| `./go refresh` | 根据服务器实时事实重建当前运行视图 |

启动实验示例：

```bash
./go run --dry-run -- python 程序入口.py --config configs/train.yaml
./go run --host gpu1 --gpu 0 --name baseline -- python 程序入口.py --epochs 100
```

如果想先确认会停止哪个远程进程，可以使用 `./go kill <run_id> --dry-run`。

查看 run 示例：

```bash
./go runs
./go runs --limit 30
./go runs --all
./go runs --json
```

拉取结果示例：

```bash
./go pull
./go pull --host gpu1
```

需要拉回哪些日志、输出或模型文件，建议在 `.remote_go/pull.yaml` 中配置。

## 面向 AI 的使用

Remote_GO 的第一目标仍然是让人用起来简单；在不增加人工使用负担的前提下，它也提供适合上层 AI 调用的稳定接口：

- `./go status --json`：读取当前服务器、GPU 和进程状态。
- `./go runs --json`：读取最近 run 的 `run_id`、服务器、GPU、命令、日志路径和状态。
- `./go refresh --json`：根据服务器实时事实重建当前运行视图。
- `./go run --dry-run -- ...` 和 `./go kill <run_id> --dry-run`：先预览远程动作，再决定是否执行。

这样 AI 不需要猜测 shell 历史、tmux 窗口、进程名或临时日志文件，而是可以通过项目本地统一接口做更准确的实验管理。

## 构建和验证

本地开发验证：

```bash
python -m unittest discover -s tests -v
```

构建可分发包：

```bash
python -m pip install build
python -m build
```

## 安全设计

- `go run` 同步到 `remote.root/releases/<run_id>/`。
- `go push` 默认同步到 `remote.root/workspace/`。
- `go pull` 只使用 `.remote_go/pull.yaml` 中的白名单规则。
- 远程路径如果逃逸出 `remote.root` 会被拒绝。
- 运行历史追加写入 `.remote_go/state/registry.jsonl`。
- `go refresh` 写入 `.remote_go/state/current.json`，不会改写历史。使用 `--preview` 可以只预览不写入。
- `go kill` 只会向 Remote_GO 能证明属于当前项目、且属于当前 SSH 用户的进程发送信号。

## License

Remote_GO 使用 MIT License。详见 [LICENSE](LICENSE)。
