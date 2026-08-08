"""可观测：节点耗时 / 调用记录 tracer。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class Tracer:
    def __init__(self, trace_file: str | Path | None = None) -> None:
        self.trace_file = Path(trace_file) if trace_file else None
        self.spans: List[Dict] = []

    def record(self, name: str, ms: float, extra: Optional[Dict] = None) -> Dict:
        rec: Dict[str, Any] = {"node": name, "ms": round(ms, 1)}
        if extra:
            rec.update(extra)
        self.spans.append(rec)
        if self.trace_file:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def summary(self) -> List[Dict]:
        return self.spans
