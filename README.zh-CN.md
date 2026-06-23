# Remote_GO

[English](README.md) | 中文

Remote_GO 是一个项目本地化的 SSH/tmux 远程实验命令工具。它用于读取远程 GPU 状态、启动自己的实验、跟踪最近运行记录，并按规则把日志或结果拉回本地。

它刻意保持轻量。Remote_GO 不是调度系统、云平台、队列系统，也不是完整实验管理平台。

## 适用场景

| 场景 | Remote_GO 能做什么 | 不覆盖什么 |
| --- | --- | --- |
| 一个本地代码项目，固定几台实验室服务器 | 把主机优先级、SSH 目标、远程根目录、conda 环境放在项目配置里 | 管理动态云机器 |
| 学生或研究者共用少量 GPU 服务器 | 查看 GPU 是空闲、被自己占用、被别人占用，还是有未跟踪进程 | 公平调度、配额、排队 |
| 日常启动实验 | 推送项目、在 tmux 中启动实验、分配稳定 `RUN_ID`、写入本地运行记录 | 替代 Slurm、Kubernetes 或 Weights & Biases |
| 查看近期实验 | 显示最近 runs、查看日志尾部、用服务器实时事实重建当前视图 | 长期指标存储 |
| 拉回结果 | 只按可编辑白名单拉回日志或输出 | 无限制镜像远程目录 |

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

克隆仓库：

```bash
git clone https://github.com/<your-name>/Remote_GO.git
cd Remote_GO
```

安装本地 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

可以直接用源码模式初始化你的项目：

```bash
./go init --project-root /path/to/your_project
```

也可以用 editable 模式安装：

```bash
python -m pip install -e .
remote-go init --project-root /path/to/your_project
```

执行 `init` 后，进入你自己的项目，用生成的 `go` 包装脚本执行命令：

```bash
cd /path/to/your_project
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
  push_target: workspace
```

通常需要填写这些字段：

- `remote.root`：每台远程服务器上的绝对路径。
- `remote.env.name`：远程 conda 环境名，不是本地 Python 环境。
- `hosts[].name`：`./go --host` 使用的短名称。
- `hosts[].ssh`：本地 `ssh` 命令可直接识别的 SSH 目标。
- `.remote_go/push.exclude`：上传时跳过的文件。
- `.remote_go/pull.yaml`：允许拉回的远程目录和文件模式。

## 命令

| 命令 | 作用 |
| --- | --- |
| `./go init` | 在项目中创建 Remote_GO 文件 |
| `./go status [--host gpu1]` | 查看配置的远程 GPU 状态 |
| `./go run -- python train.py` | 推送代码并启动远程 tmux 实验 |
| `./go runs [--limit 30]` | 查看最近运行记录，默认显示 12 条 |
| `./go log <run_id>` | 查看某次远程实验的日志尾部 |
| `./go kill <run_id> --dry-run` | 预览停止自己的某个 tracked run |
| `./go kill <run_id> --yes` | 停止一个已确认的 Remote_GO run |
| `./go push [--host gpu1]` | 推送项目文件到 `remote.root/workspace/` |
| `./go pull --kind logs` | 把允许的日志或输出拉回本地 |
| `./go refresh --apply` | 根据服务器实时事实重建当前运行视图 |

启动实验示例：

```bash
./go run --dry-run -- python train.py --config configs/train.yaml
./go run --host gpu1 --gpu 0 --name baseline -- python train.py --epochs 100
```

查看 run 示例：

```bash
./go runs
./go runs --limit 30
./go runs --all
```

拉取结果示例：

```bash
./go pull --kind logs
./go pull --kind outputs --host gpu1
```

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
- `go refresh --apply` 写入 `.remote_go/state/current.json`，不会改写历史。
- `go kill` 只会向 Remote_GO 能证明属于当前项目、且属于当前 SSH 用户的进程发送信号。

## License

Remote_GO 使用 MIT License。详见 [LICENSE](LICENSE)。
