"""Small I/O helpers: provenance, logging, check bookkeeping."""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np


class RunLogger:
    """Timestamped logger that echoes to stdout and to a run log file."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_header(path: Path) -> list[str]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))


def file_record(path: Path, role: str, used: bool) -> dict:
    """Provenance record. Raw file is opened read-only; never modified."""
    st = path.stat()
    with open(path, "rb") as f:
        starts_with_bom = f.read(3) == b"\xef\xbb\xbf"
    header = read_header(path)
    return {
        "role": role,
        "used_in_pipeline": used,
        "filename": path.name,
        "absolute_path": str(path.resolve()),
        "size_bytes": st.st_size,
        "last_modified_local": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
        "utf8_bom": starts_with_bom,
        "n_columns": len(header),
        "column_names": header,
    }


def environment_record() -> dict:
    import pandas, pyarrow  # noqa: E401
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pandas.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "run_started": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable, ensure_ascii=False)


class CheckRegistry:
    """Collects every audit check, including those with zero failures."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, check_id: str, stage: str, description: str, n_checked: int,
            n_failures: int, *, severity: str = "error", status: str | None = None,
            notes: str = "") -> None:
        n_checked = int(n_checked)
        n_failures = int(n_failures)
        if status is None:
            if severity == "info":
                status = "INFO"
            else:
                status = "PASS" if n_failures == 0 else ("WARN" if severity == "warning" else "FAIL")
        self.rows.append({
            "check_id": check_id, "stage": stage, "description": description,
            "n_checked": n_checked, "n_failures": n_failures,
            "severity": severity, "status": status, "notes": notes,
        })

    def get(self, check_id: str) -> dict | None:
        for r in self.rows:
            if r["check_id"] == check_id:
                return r
        return None

    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]
