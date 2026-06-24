from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


REMOTE_GO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REMOTE_GO_ROOT))

from remote_go import config  # noqa: E402
from remote_go.core import console, run_world  # noqa: E402
from remote_go.core.adapter import ProjectAdapter, PullSpec, TaskSpec  # noqa: E402


class RemoteGoTests(unittest.TestCase):
    maxDiff = None

    def make_project(self) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = Path(tmp_dir.name) / "demo_project"
        root.mkdir()
        (root / "train.py").write_text("print('train')\n")
        config.init_project(root)
        cfg = config.read_yaml(root / ".remote_go" / "config.yaml")
        cfg["project"]["id"] = "demo"
        cfg["project"]["label"] = "Demo"
        cfg["remote"]["root"] = "/remote/demo"
        cfg["remote"]["env"]["name"] = "pytorch_env"
        cfg["hosts"] = [
            {"name": "gpu1", "ssh": "user@gpu1", "idle_mem_threshold_mib": 100, "idle_util_threshold_percent": 8},
            {"name": "gpu2", "ssh": "user@gpu2", "idle_mem_threshold_mib": 100, "idle_util_threshold_percent": 8},
        ]
        config.write_yaml(root / ".remote_go" / "config.yaml", cfg)
        return root

    def run_go(self, root: Path, args: list[str], expect_success: bool = True) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REMOTE_GO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "remote_go.cli", *args],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if expect_success and proc.returncode != 0:
            self.fail(f"command failed: {args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        if not expect_success and proc.returncode == 0:
            self.fail(f"command unexpectedly passed: {args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        return proc

    def load_adapter(self, root: Path) -> ProjectAdapter:
        adapter, _ = config.load_project(root)
        return adapter

    def test_init_generates_project_files_and_gitignore_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "paper_code"
            root.mkdir()
            created = config.init_project(root)

            self.assertTrue((root / "go").exists())
            self.assertTrue(os.access(root / "go", os.X_OK))
            self.assertTrue((root / ".remote_go" / "config.yaml").exists())
            self.assertTrue((root / ".remote_go" / "push.exclude").exists())
            self.assertTrue((root / ".remote_go" / "pull.yaml").exists())
            self.assertTrue((root / ".remote_go" / "gitignore.block").exists())
            self.assertTrue((root / ".remote_go" / "state" / "registry.jsonl").exists())
            self.assertIn(".remote_go/state/", (root / ".gitignore").read_text())
            self.assertTrue(any(path.name == "config.yaml" for path in created))

    def test_load_project_maps_simple_yaml_to_adapter(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)

        self.assertEqual(adapter.project_id, "demo")
        self.assertEqual(adapter.project_label, "Demo")
        self.assertEqual(adapter.remote_project_root, "/remote/demo")
        self.assertEqual(adapter.conda_env, "pytorch_env")
        self.assertEqual(adapter.default_task, "command")
        self.assertEqual(adapter.build_command("command", ["--", "python", "train.py"]), ["python", "train.py"])
        self.assertEqual(adapter.local_registry, (root / ".remote_go" / "state" / "registry.jsonl").resolve())
        self.assertIn(".git/", adapter.rsync_exclude_patterns)
        self.assertEqual({spec.name for spec in adapter.pull_specs}, {"logs", "outputs"})
        outputs = next(spec for spec in adapter.pull_specs if spec.name == "outputs")
        self.assertIn("*.ckpt", outputs.include_patterns)

    def test_hosts_order_is_preserved_as_priority(self) -> None:
        root = self.make_project()
        hosts = console.load_hosts(root / ".remote_go" / "config.yaml")
        self.assertEqual([host.name for host in hosts], ["gpu1", "gpu2"])

    def test_status_normalization_separates_ours_external_and_untracked(self) -> None:
        adapter = self.load_adapter(self.make_project())
        host = console.HostConfig(name="gpu1", ssh="user@gpu1")

        idle = console.normalize_gpu_status(adapter, host, {
            "index": 0,
            "memory_used_mib": 0,
            "memory_total_mib": 1000,
            "utilization_gpu": 0,
            "processes": [],
        })
        ours = console.normalize_gpu_status(adapter, host, {
            "index": 0,
            "memory_used_mib": 500,
            "memory_total_mib": 1000,
            "utilization_gpu": 20,
            "processes": [{"run_id": "run1", "project": "Demo", "comment": "baseline"}],
        })
        untracked = console.normalize_gpu_status(adapter, host, {
            "index": 0,
            "memory_used_mib": 500,
            "memory_total_mib": 1000,
            "utilization_gpu": 20,
            "processes": [{"pid": 123, "is_current_user": True}],
        })
        external = console.normalize_gpu_status(adapter, host, {
            "index": 0,
            "memory_used_mib": 500,
            "memory_total_mib": 1000,
            "utilization_gpu": 20,
            "processes": [{"pid": 456, "is_current_user": False}],
        })

        self.assertEqual(idle["state"], "idle")
        self.assertEqual(ours["state"], "busy_ours")
        self.assertEqual(untracked["state"], "busy_ours_untracked")
        self.assertEqual(external["state"], "busy_external")
        self.assertIn("comment:baseline", ours["note"])

    def test_go_run_dry_run_uses_user_command_and_local_registry(self) -> None:
        root = self.make_project()
        proc = self.run_go(root, ["run", "--dry-run", "--host", "gpu1", "--gpu", "0", "--", "python", "train.py", "--epochs", "1"])
        payload = json.loads(proc.stdout)

        self.assertEqual(payload["project"], "demo")
        self.assertEqual(payload["selected"], {"host": "gpu1", "gpu": 0})
        self.assertEqual(payload["command"], ["conda", "run", "--no-capture-output", "-n", "pytorch_env", "python", "train.py", "--epochs", "1"])
        self.assertEqual(payload["registry"], str((root / ".remote_go" / "state" / "registry.jsonl").resolve()))
        self.assertTrue(payload["release_dir"].startswith("/remote/demo/releases/demo_"))
        self.assertIn("--exclude", payload["sync_preview"]["command"])

    def test_go_run_requires_command_after_separator(self) -> None:
        root = self.make_project()
        proc = self.run_go(root, ["run", "--dry-run"], expect_success=False)
        self.assertIn("requires a command", proc.stderr)

    def test_launch_tmux_pane_uses_stable_window_id(self) -> None:
        adapter = self.load_adapter(self.make_project())
        host = console.HostConfig(name="gpu1", ssh="user@gpu1")
        captured = {}

        def fake_run_ssh(host_arg, remote_command, capture=True, check=False):
            captured["host"] = host_arg
            captured["remote_command"] = remote_command
            return subprocess.CompletedProcess(["ssh"], 0, "", "")

        with mock.patch.object(console, "run_ssh", fake_run_ssh):
            console.launch_tmux_pane(adapter, host, "/remote/demo/runs/run1/pane.sh", 0, "run1")

        remote_command = captured["remote_command"]
        self.assertIn("window_id=", remote_command)
        self.assertIn("automatic-rename off", remote_command)
        self.assertIn('target="$window_id"', remote_command)
        self.assertIn('tmux list-panes -t "$target"', remote_command)
        self.assertIn('tmux split-window -t "$target"', remote_command)
        self.assertNotIn("tmux list-panes -t M:M", remote_command)
        self.assertNotIn("grep -Fxq", remote_command)

    def test_go_run_clears_reservation_and_skips_registry_when_initial_status_missing(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)
        args = argparse.Namespace(
            task="command",
            hosts_config=root / ".remote_go" / "config.yaml",
            host="gpu1",
            gpu=0,
            mode=None,
            dry_run=False,
            name=None,
            comment=None,
            change_note=None,
            framework_args=["--", "python", "train.py"],
        )

        with mock.patch.object(console, "query_all_status", return_value=[]), mock.patch.object(
            console, "choose_idle_gpu", return_value={"host": "gpu1", "gpu": 0}
        ), mock.patch.object(console, "ensure_remote_dirs"), mock.patch.object(
            console, "sync_project", return_value={}
        ), mock.patch.object(
            console, "sync_required_artifacts", return_value=[]
        ), mock.patch.object(
            console, "upload_file"
        ), mock.patch.object(
            console, "run_ssh", return_value=subprocess.CompletedProcess(["ssh"], 0, "", "")
        ), mock.patch.object(
            console, "launch_tmux_pane"
        ), mock.patch.object(
            console, "wait_for_remote_run_status", return_value=None
        ), mock.patch.object(
            console, "clear_launch_reservation", return_value=True
        ) as clear_reservation:
            with self.assertRaisesRegex(RuntimeError, "did not create status.json"):
                console.command_run(args, adapter)

        clear_reservation.assert_called_once()
        self.assertEqual(clear_reservation.call_args.args[1].name, "gpu1")
        self.assertEqual(clear_reservation.call_args.args[2], 0)
        self.assertEqual(adapter.local_registry.read_text(), "")

    def test_push_dry_run_defaults_to_workspace_target(self) -> None:
        root = self.make_project()
        proc = self.run_go(root, ["push", "--dry-run"])
        payload = json.loads(proc.stdout)

        self.assertEqual(payload["push_targets"][0]["target"], "user@gpu1:/remote/demo/workspace/")
        self.assertTrue(payload["push_targets"][0]["dry_run"])

    def test_pull_dry_run_uses_allow_list_patterns(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)
        captured = {}

        def fake_run_command(cmd, capture=True, check=False):
            captured["cmd"] = list(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        args = argparse.Namespace(
            hosts_config=root / ".remote_go" / "config.yaml",
            host="gpu1",
            kind="logs",
            dry_run=True,
        )
        output = io.StringIO()
        with mock.patch.object(console, "run_command", fake_run_command), redirect_stdout(output):
            console.command_pull(args, adapter)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["pulled"][0]["remote_dir"], "/remote/demo/logs")
        self.assertIn("--include", captured["cmd"])
        self.assertIn("*.log", captured["cmd"])
        self.assertIn("--exclude", captured["cmd"])

    def test_runs_local_only_default_limit_is_twelve(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)
        for index in range(20):
            console.append_registry(adapter.local_registry, {
                "project_id": "demo",
                "run_id": f"run-{index}",
                "created_at": f"2026-01-01T00:{index:02d}:00+00:00",
                "task": "command",
                "mode": "command",
                "host": "gpu1",
                "gpu": 0,
                "release_dir": f"/remote/demo/releases/run-{index}",
                "log_file": f"/remote/demo/logs/remote_go/run-{index}.log",
                "command": ["python", "train.py"],
                "remote_status": {"state": "COMPLETED"},
            })

        proc = self.run_go(root, ["runs", "--local-only"])
        self.assertNotIn("run-7", proc.stdout)
        self.assertIn("run-8", proc.stdout)
        self.assertIn("run-19", proc.stdout)

    def test_runs_json_returns_structured_records(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)
        console.append_registry(adapter.local_registry, {
            "project_id": "demo",
            "run_id": "run-json",
            "created_at": "2026-01-01T00:00:00+00:00",
            "task": "command",
            "mode": "command",
            "host": "gpu1",
            "gpu": 0,
            "release_dir": "/remote/demo/releases/run-json",
            "log_file": "/remote/demo/logs/remote_go/run-json.log",
            "command": ["python", "train.py"],
            "remote_status": {"state": "COMPLETED"},
        })

        proc = self.run_go(root, ["runs", "--local-only", "--json"])
        payload = json.loads(proc.stdout)

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["records"][0]["run_id"], "run-json")
        self.assertEqual(payload["records"][0]["host"], "gpu1")

    def test_run_world_current_view_keeps_all_running_rows_over_limit(self) -> None:
        records = [
            {"run_id": f"done-{index}", "remote_status": {"state": "COMPLETED"}}
            for index in range(10)
        ] + [
            {"run_id": f"live-{index}", "remote_status": {"state": "RUNNING"}}
            for index in range(6)
        ]

        limited = run_world.limit_current_records(records, limit=3)

        self.assertEqual([record["run_id"] for record in limited], [f"live-{index}" for index in range(6)])

    def test_refresh_default_writes_current_snapshot_without_rewriting_registry(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)
        console.append_registry(adapter.local_registry, {
            "project_id": "demo",
            "project_label": "Demo",
            "run_id": "registered-live",
            "created_at": "2026-01-01T00:00:00+00:00",
            "task": "command",
            "mode": "command",
            "host": "gpu1",
            "gpu": 0,
            "release_dir": "/remote/demo/releases/registered-live",
            "log_file": "/remote/demo/logs/remote_go/registered-live.log",
            "command": ["python", "train.py"],
        })
        registry_before = adapter.local_registry.read_text()

        status_payloads = [{
            "project_id": "demo",
            "project_label": "Demo",
            "host": "gpu1",
            "gpus": [{
                "host": "gpu1",
                "gpu": 0,
                "state": "busy_ours",
                "tmux": "M:M",
                "processes": [{"pid": 1, "is_current_user": True, "run_id": "registered-live", "run_id_source": "env"}],
            }, {
                "host": "gpu1",
                "gpu": 1,
                "state": "busy_ours",
                "tmux": "M:M",
                "processes": [{"pid": 2, "is_current_user": True, "run_id": "live-only", "run_id_source": "cwd"}],
            }],
        }]

        args = argparse.Namespace(
            hosts_config=root / ".remote_go" / "config.yaml",
            host=None,
            json=True,
            limit=12,
            verbose=False,
        )
        with mock.patch.object(console, "query_all_status", return_value=status_payloads), mock.patch.object(
            console, "fetch_remote_run_status", return_value={"state": "RUNNING"}
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                console.command_refresh(args, [adapter])

        payload = json.loads(output.getvalue())
        current_path = root / ".remote_go" / "state" / "current.json"
        self.assertTrue(current_path.exists())
        self.assertEqual(adapter.local_registry.read_text(), registry_before)
        self.assertTrue(payload["applied"])
        self.assertEqual([record["run_id"] for record in payload["records"]], ["registered-live", "live-only"])

    def test_kill_candidates_require_current_project_authorization(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)
        records = [{
            "project_id": "demo",
            "run_id": "demo-live",
            "host": "gpu1",
            "gpu": 0,
        }]
        status_payloads = [{
            "host": "gpu1",
            "gpus": [{
                "gpu": 0,
                "processes": [{
                    "pid": 11,
                    "is_current_user": True,
                    "run_id": "demo-live",
                    "project": "Demo",
                    "cwd": "/remote/demo/releases/demo-live",
                }, {
                    "pid": 12,
                    "is_current_user": True,
                    "run_id": "foreign-live",
                    "project": "",
                    "cwd": "/some/other/project",
                }, {
                    "pid": 13,
                    "is_current_user": False,
                    "run_id": "demo-live",
                    "project": "Demo",
                    "cwd": "/remote/demo/releases/demo-live",
                }],
            }],
        }]

        allowed = console.find_kill_candidates(adapter, status_payloads, records, "demo-live")
        blocked = console.find_kill_candidates(adapter, status_payloads, records, "foreign-live")

        self.assertEqual([candidate["pid"] for candidate in allowed], [11])
        self.assertEqual(blocked, [])

    def test_kill_sends_signal_after_authorization_without_yes(self) -> None:
        root = self.make_project()
        adapter = self.load_adapter(root)
        args = argparse.Namespace(
            hosts_config=root / ".remote_go" / "config.yaml",
            key="demo-live",
            host=None,
            gpu=None,
            signal="TERM",
            dry_run=False,
            yes=False,
            all=False,
        )
        status_payloads = [{
            "host": "gpu1",
            "gpus": [{
                "gpu": 0,
                "processes": [{
                    "pid": 11,
                    "is_current_user": True,
                    "run_id": "demo-live",
                    "project": "Demo",
                    "cwd": "/remote/demo/releases/demo-live",
                }],
            }],
        }]
        console.append_registry(adapter.local_registry, {
            "project_id": "demo",
            "run_id": "demo-live",
            "host": "gpu1",
            "gpu": 0,
            "task": "command",
            "mode": "command",
            "release_dir": "/remote/demo/releases/demo-live",
            "log_file": "/remote/demo/logs/remote_go/demo-live.log",
            "command": ["python", "train.py"],
        })

        output = io.StringIO()
        with mock.patch.object(console, "query_all_status", return_value=status_payloads), mock.patch.object(
            console, "kill_remote_process", return_value={"ok": True, "dry_run": False, "pid": 11}
        ) as kill_mock:
            with redirect_stdout(output):
                console.command_kill(args, adapter)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["dry_run"])
        kill_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
