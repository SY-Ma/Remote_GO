from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def shorten_run_id(run_id: str) -> str:
    if not run_id:
        return ""
    if len(run_id) <= 18:
        return run_id
    return f"{run_id[:15]}...{run_id[-8:]}"


def append_note(existing: str, note: str) -> str:
    if not note:
        return existing
    if not existing:
        return note
    if note in existing.split("; "):
        return existing
    return f"{existing}; {note}"


def limit_run_records(records: Sequence[Dict[str, Any]], limit: int, all_records: bool = False) -> List[Dict[str, Any]]:
    if all_records:
        return list(records)
    if limit <= 0:
        raise ValueError("--limit must be positive, or use --all-records.")
    return list(records)[-limit:]


def limit_current_records(records: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--limit must be positive, or use --all-records.")
    running = [
        record
        for record in records
        if str(record.get("remote_status", {}).get("state", "")).upper() == "RUNNING"
    ]
    non_running = [
        record
        for record in records
        if str(record.get("remote_status", {}).get("state", "")).upper() != "RUNNING"
    ]
    remaining = max(0, limit - len(running))
    if remaining == 0:
        return running
    return non_running[-remaining:] + running


def display_run_state(remote_status: Dict[str, Any]) -> str:
    return str(remote_status.get("state", "UNKNOWN") or "UNKNOWN")


@dataclass
class RunWorld:
    records: List[Dict[str, Any]]
    status_payloads: List[Dict[str, Any]]
    known_run_ids: set[str]
    live_by_run_id: Dict[str, Dict[str, Any]]
    live_locations_by_run_id: Dict[str, List[Dict[str, Any]]]
    gpu_by_host: Dict[Tuple[str, int], Dict[str, Any]]

    @classmethod
    def from_sources(
        cls,
        records: Sequence[Dict[str, Any]],
        status_payloads: Sequence[Dict[str, Any]],
        known_run_ids: Optional[set[str]] = None,
    ) -> "RunWorld":
        live_by_run_id, live_locations_by_run_id, gpu_by_host = live_status_indexes(status_payloads)
        if known_run_ids is None:
            known_run_ids = {str(record.get("run_id")) for record in records if record.get("run_id")}
        return cls(
            records=list(records),
            status_payloads=list(status_payloads),
            known_run_ids=known_run_ids,
            live_by_run_id=live_by_run_id,
            live_locations_by_run_id=live_locations_by_run_id,
            gpu_by_host=gpu_by_host,
        )

    def annotated_records(self) -> List[Dict[str, Any]]:
        return annotate_records_with_live_status(
            self.records,
            self.status_payloads,
            live_by_run_id=self.live_by_run_id,
            live_locations_by_run_id=self.live_locations_by_run_id,
            gpu_by_host=self.gpu_by_host,
        )

    def effective_records(self) -> List[Dict[str, Any]]:
        return filter_effective_run_records(
            self.annotated_records(),
            live_by_run_id=self.live_by_run_id,
        )

    def live_only_records(self) -> List[Dict[str, Any]]:
        return build_live_only_records(
            self.status_payloads,
            known_run_ids=self.known_run_ids,
            known_locations_by_run_id=record_locations_by_run_id(self.records),
        )

    def current_records(self) -> List[Dict[str, Any]]:
        return self.effective_records() + self.live_only_records()

    def status_anomalies(self) -> List[Dict[str, Any]]:
        return collect_status_anomalies(self.status_payloads, known_run_ids=self.known_run_ids)

    def lifecycle_anomalies(self) -> List[Dict[str, Any]]:
        return collect_run_lifecycle_anomalies(self.annotated_records())

    def anomalies(self) -> List[Dict[str, Any]]:
        return self.status_anomalies() + self.lifecycle_anomalies()

    def missing_live_locations_from_current_records(self, current_records: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
        return missing_live_locations_from_records(current_records, self.status_payloads)


def live_status_indexes(
    status_payloads: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, int], Dict[str, Any]]]:
    live_by_run_id: Dict[str, Dict[str, Any]] = {}
    live_locations_by_run_id: Dict[str, List[Dict[str, Any]]] = {}
    gpu_by_host: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for payload in status_payloads:
        if payload.get("error"):
            continue
        host = payload.get("host", "")
        for gpu in payload.get("gpus", []):
            gpu_index = int(gpu.get("gpu", -1))
            processes = gpu.get("processes", [])
            tracked_run_ids = sorted({proc.get("run_id") for proc in processes if proc.get("run_id")})
            untracked_own = [proc for proc in processes if proc.get("is_current_user") and not proc.get("run_id")]
            external = [proc for proc in processes if not proc.get("run_id") and not proc.get("is_current_user")]
            gpu_by_host[(host, gpu_index)] = {
                "state": gpu.get("state", ""),
                "tracked_run_ids": tracked_run_ids,
                "untracked_own_count": len(untracked_own),
                "external_count": len(external),
            }
            for proc in processes:
                run_id = proc.get("run_id")
                if run_id:
                    location = {
                        "project": payload.get("project_label", payload.get("project_id", "")),
                        "host": host,
                        "gpu": gpu_index,
                        "process": proc,
                    }
                    live_by_run_id.setdefault(str(run_id), location)
                    live_locations_by_run_id.setdefault(str(run_id), []).append(location)
    return live_by_run_id, live_locations_by_run_id, gpu_by_host


def annotate_records_with_live_status(
    records: Sequence[Dict[str, Any]],
    status_payloads: Sequence[Dict[str, Any]],
    live_by_run_id: Optional[Dict[str, Dict[str, Any]]] = None,
    live_locations_by_run_id: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    gpu_by_host: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if live_by_run_id is None or live_locations_by_run_id is None or gpu_by_host is None:
        live_by_run_id, live_locations_by_run_id, gpu_by_host = live_status_indexes(status_payloads)
    running_by_gpu: Dict[Tuple[str, int], List[str]] = {}
    for record in records:
        remote_status = record.get("remote_status", {})
        if str(remote_status.get("state", "")).upper() != "RUNNING":
            continue
        try:
            gpu = int(record.get("gpu"))
        except (TypeError, ValueError):
            continue
        running_by_gpu.setdefault((str(record.get("host", "")), gpu), []).append(str(record.get("run_id", "")))

    annotated = []
    for record in records:
        row = dict(record)
        remote_status = dict(row.get("remote_status", {}))
        run_id = str(row.get("run_id", ""))
        state = str(remote_status.get("state", "")).upper()
        note = str(remote_status.get("note", "") or "")
        try:
            gpu = int(row.get("gpu"))
        except (TypeError, ValueError):
            gpu = -1
        host_gpu = (str(row.get("host", "")), gpu)

        conflicts = [item for item in running_by_gpu.get(host_gpu, []) if item]
        if state == "RUNNING" and len(conflicts) > 1:
            note = append_note(note, f"running_registry_conflict:{len(conflicts)}")

        live_locations = live_locations_by_run_id.get(run_id, [])
        live_location = live_locations[0] if live_locations else None
        live_gpu = gpu_by_host.get(host_gpu)
        if state == "RUNNING":
            if live_location is None:
                if live_gpu:
                    if live_gpu["untracked_own_count"]:
                        note = append_note(note, "run_id_not_seen;gpu_has_untracked_ours")
                    elif live_gpu["tracked_run_ids"]:
                        visible = ",".join(shorten_run_id(item) for item in live_gpu["tracked_run_ids"][:2])
                        note = append_note(note, f"run_id_not_seen;gpu_live_run:{visible}")
                    elif live_gpu["external_count"]:
                        note = append_note(note, "run_id_not_seen;gpu_busy_external")
                    elif live_gpu["state"] == "idle":
                        note = append_note(note, "run_id_not_seen;gpu_idle")
                    else:
                        note = append_note(note, f"run_id_not_seen;gpu_state:{live_gpu['state']}")
                else:
                    note = append_note(note, "run_id_not_seen;live_status_unavailable")
            elif not any(location.get("host") == row.get("host") and location.get("gpu") == gpu for location in live_locations):
                note = append_note(
                    note,
                    f"live_location_mismatch:{live_location.get('host')}:{live_location.get('gpu')}",
                )
            if len(live_locations) > 1:
                note = append_note(note, f"duplicate_live_run_id:{len(live_locations)}")
        elif state in {"COMPLETED", "FAILED"}:
            if live_location is not None:
                note = append_note(note, "finished_status_but_live_process")

        if note:
            remote_status["note"] = note
        row["remote_status"] = remote_status
        annotated.append(row)
    return annotated


def record_gpu_key(record: Dict[str, Any]) -> Tuple[str, str, int]:
    try:
        gpu = int(record.get("gpu"))
    except (TypeError, ValueError):
        gpu = -1
    return str(record.get("project_id", "")), str(record.get("host", "")), gpu


def record_locations_by_run_id(records: Sequence[Dict[str, Any]]) -> Dict[str, set[Tuple[str, int]]]:
    locations: Dict[str, set[Tuple[str, int]]] = {}
    for record in records:
        run_id = str(record.get("run_id", ""))
        if not run_id:
            continue
        try:
            gpu = int(record.get("gpu"))
        except (TypeError, ValueError):
            continue
        locations.setdefault(run_id, set()).add((str(record.get("host", "")), gpu))
    return locations


def live_run_location_keys(status_payloads: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    keys = []
    for payload in status_payloads:
        if payload.get("error"):
            continue
        host = str(payload.get("host", ""))
        for gpu in payload.get("gpus", []):
            try:
                gpu_index = int(gpu.get("gpu", -1))
            except (TypeError, ValueError):
                gpu_index = -1
            for process in gpu.get("processes", []):
                run_id = str(process.get("run_id") or "")
                if process.get("is_current_user") and run_id:
                    keys.append((run_id, host, gpu_index))
    return keys


def run_record_location_keys(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    keys = []
    for record in records:
        run_id = str(record.get("run_id", ""))
        if not run_id:
            continue
        try:
            gpu = int(record.get("gpu"))
        except (TypeError, ValueError):
            continue
        keys.append((run_id, str(record.get("host", "")), gpu))
    return keys


def missing_live_locations_from_records(
    records: Sequence[Dict[str, Any]],
    status_payloads: Sequence[Dict[str, Any]],
) -> List[Tuple[str, str, int]]:
    record_keys = set(run_record_location_keys(records))
    return [key for key in live_run_location_keys(status_payloads) if key not in record_keys]


def filter_effective_run_records(
    records: Sequence[Dict[str, Any]],
    live_by_run_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    live_records_by_gpu: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for record in records:
        run_id = str(record.get("run_id", ""))
        if run_id not in live_by_run_id:
            continue
        live_records_by_gpu[record_gpu_key(record)] = record

    effective = []
    for record in records:
        run_id = str(record.get("run_id", ""))
        remote_status = record.get("remote_status", {})
        state = str(remote_status.get("state", "")).upper()
        note = str(remote_status.get("note", "") or "")
        if run_id in live_by_run_id:
            effective.append(record)
            continue
        if state == "RUNNING" and "run_id_not_seen" in note and "gpu_live_run:" in note:
            continue
        live_record = live_records_by_gpu.get(record_gpu_key(record))
        if state == "FAILED" and live_record is not None:
            record_created = str(record.get("created_at", ""))
            live_created = str(live_record.get("created_at", ""))
            if not record_created or not live_created or live_created >= record_created:
                continue
        effective.append(record)
    return effective


def build_live_only_records(
    status_payloads: Sequence[Dict[str, Any]],
    known_run_ids: set[str],
    known_locations_by_run_id: Optional[Dict[str, set[Tuple[str, int]]]] = None,
) -> List[Dict[str, Any]]:
    records = []
    known_locations_by_run_id = known_locations_by_run_id or {}
    for payload in status_payloads:
        if payload.get("error"):
            continue
        for gpu in payload.get("gpus", []):
            for process in gpu.get("processes", []):
                run_id = str(process.get("run_id") or "")
                if not run_id:
                    continue
                if not process.get("is_current_user"):
                    continue
                host = str(gpu.get("host", payload.get("host", "")))
                try:
                    gpu_index = int(gpu.get("gpu", -1))
                except (TypeError, ValueError):
                    gpu_index = -1
                known_locations = known_locations_by_run_id.get(run_id, set())
                if run_id in known_run_ids and (host, gpu_index) in known_locations:
                    continue
                note = "live_location_not_in_registry" if run_id in known_run_ids else "live_only_missing_registry"
                if process.get("run_id_source"):
                    note = append_note(note, f"run_id_from_{process['run_id_source']}")
                if process.get("cwd"):
                    note = append_note(note, "cwd_seen")
                records.append({
                    "project_id": payload.get("project_id", ""),
                    "project_label": payload.get("project_label", payload.get("project_id", "")),
                    "run_id": run_id,
                    "created_at": "LIVE",
                    "task": "live",
                    "mode": "unregistered",
                    "comment": process.get("comment", ""),
                    "change_note": process.get("change_note", ""),
                    "host": host,
                    "gpu": gpu_index,
                    "tmux_session": "M" if gpu.get("tmux") else "",
                    "tmux_window": "",
                    "release_dir": process.get("cwd", ""),
                    "log_file": "",
                    "command": [process.get("process_name", "")] if process.get("process_name") else [],
                    "remote_status": {
                        "state": "RUNNING",
                        "exit_code": 0,
                        "note": note,
                    },
                    "live_only": True,
                })
    return records


def collect_status_anomalies(
    status_payloads: Sequence[Dict[str, Any]],
    known_run_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    anomalies = []
    _, live_locations_by_run_id, _ = live_status_indexes(status_payloads)
    anomaly_actions = {
        "busy_ours_untracked": (
            "current_user_gpu_process_missing_remotecontrol_identity",
            "确认该进程是否必须继续；后续同类任务必须通过 Remote_GO 启动以获得 run_id。",
        ),
        "busy_ours_conflict": (
            "multiple_own_processes_on_one_gpu",
            "按 run_id / pid / cwd 人工确认；一卡一程序原则下不应再启动新任务。",
        ),
        "busy_mixed": (
            "own_and_external_processes_share_gpu",
            "不要启动新任务；先确认是否属于共享 GPU 或异常占用。",
        ),
    }
    for payload in status_payloads:
        if payload.get("error"):
            anomalies.append({
                "type": "host_status_error",
                "severity": "warning",
                "project": payload.get("project_label", payload.get("project_id", "")),
                "host": payload.get("host", ""),
                "gpu": "-",
                "state": "unknown",
                "run_id": "",
                "note": str(payload.get("error", "")),
                "action": "先修复 status 查询能力；不要基于该 host 的未知状态启动任务。",
            })
            continue
        for gpu in payload.get("gpus", []):
            state = str(gpu.get("state", ""))
            if state in anomaly_actions:
                issue_type, action = anomaly_actions[state]
                anomalies.append({
                    "type": issue_type,
                    "severity": "error",
                    "project": payload.get("project_label", payload.get("project_id", "")),
                    "host": gpu.get("host", payload.get("host", "")),
                    "gpu": str(gpu.get("gpu", "")),
                    "state": state,
                    "run_id": gpu.get("run_id", ""),
                    "note": gpu.get("note", ""),
                    "action": action,
                })
            if known_run_ids is None:
                continue
            for run_id in sorted({proc.get("run_id") for proc in gpu.get("processes", []) if proc.get("run_id")}):
                if str(run_id) in known_run_ids:
                    continue
                anomalies.append({
                    "type": "live_run_missing_registry",
                    "severity": "error",
                    "project": payload.get("project_label", payload.get("project_id", "")),
                    "host": gpu.get("host", payload.get("host", "")),
                    "gpu": str(gpu.get("gpu", "")),
                    "state": state,
                    "run_id": str(run_id),
                    "note": "live process has run_id but no central/local registry record",
                    "action": "确认是否为手动/旧流程启动；正式实验应补齐 registry 或用 Remote_GO 重新启动。",
                })
    for run_id, locations in sorted(live_locations_by_run_id.items()):
        unique_locations = sorted({(str(item.get("host", "")), str(item.get("gpu", ""))) for item in locations})
        if len(unique_locations) <= 1:
            continue
        project = str(locations[0].get("project", "")) if locations else ""
        anomalies.append({
            "type": "duplicate_live_run_id",
            "severity": "error",
            "project": project,
            "host": ",".join(f"{host}:{gpu}" for host, gpu in unique_locations),
            "gpu": "-",
            "state": "RUNNING",
            "run_id": run_id,
            "note": "same run_id appears in multiple live locations",
            "action": "run_id 必须唯一；先按 host/GPU/pid/cwd/log 区分真实实验，再补 registry 或重命名/标注异常记录。",
        })
    return anomalies


def collect_run_lifecycle_anomalies(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anomalies = []
    for record in records:
        remote_status = record.get("remote_status", {})
        state = str(remote_status.get("state", "")).upper()
        note = str(remote_status.get("note", "") or "")
        if state == "RUNNING" and "run_id_not_seen" in note:
            anomalies.append({
                "type": "running_status_without_live_process",
                "severity": "error",
                "project": record.get("project_label") or record.get("project_id", ""),
                "host": record.get("host", ""),
                "gpu": str(record.get("gpu", "")),
                "state": state,
                "run_id": record.get("run_id", ""),
                "note": note,
                "action": "status.json 未最终落盘或进程已消失；先查 log/status 文件，不要把该 run 当作正在运行。",
            })
        if "running_registry_conflict" in note:
            anomalies.append({
                "type": "multiple_running_records_same_gpu",
                "severity": "error",
                "project": record.get("project_label") or record.get("project_id", ""),
                "host": record.get("host", ""),
                "gpu": str(record.get("gpu", "")),
                "state": state,
                "run_id": record.get("run_id", ""),
                "note": note,
                "action": "registry/status lifecycle 存在同 GPU 多条 RUNNING 记录；需要人工核对并修正历史记录。",
            })
        if "finished_status_but_live_process" in note:
            anomalies.append({
                "type": "finished_status_but_live_process",
                "severity": "error",
                "project": record.get("project_label") or record.get("project_id", ""),
                "host": record.get("host", ""),
                "gpu": str(record.get("gpu", "")),
                "state": state,
                "run_id": record.get("run_id", ""),
                "note": note,
                "action": "同一 run_id 有 finished status 但进程仍在；必须先查 log 和进程命令确认。",
            })
    return anomalies
