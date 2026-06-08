"""Request pipeline timing — logs per-stage latency and sequential vs parallel flow."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_current: ContextVar[PipelineTiming | None] = ContextVar("pipeline_timing", default=None)


@dataclass
class StageRecord:
    name: str
    duration_ms: float
    mode: str  # "sequential" | "parallel"
    parallel_group: str | None = None
    order: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class PipelineTiming:
    def __init__(self, *, request_id: str = "", session_id: str = "") -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.stages: list[StageRecord] = []
        self._order = 0
        self._started = time.perf_counter()

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000

    def record(
        self,
        name: str,
        duration_ms: float,
        *,
        mode: str = "sequential",
        parallel_group: str | None = None,
        **extra: Any,
    ) -> None:
        self._order += 1
        self.stages.append(
            StageRecord(
                name=name,
                duration_ms=round(duration_ms, 2),
                mode=mode,
                parallel_group=parallel_group,
                order=self._order,
                extra=dict(extra) if extra else {},
            )
        )

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        mode: str = "sequential",
        parallel_group: str | None = None,
        **extra: Any,
    ) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                name,
                (time.perf_counter() - start) * 1000,
                mode=mode,
                parallel_group=parallel_group,
                **extra,
            )

    def run_parallel(
        self,
        group_name: str,
        tasks: dict[str, Callable[[], T]],
        *,
        max_workers: int | None = None,
    ) -> dict[str, T]:
        """Run callables concurrently; each task is logged as parallel within group_name."""
        if not tasks:
            return {}

        workers = max_workers or min(len(tasks), 4)
        results: dict[str, T] = {}
        wall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._run_parallel_task, group_name, name, fn): name
                       for name, fn in tasks.items()}
            for fut in as_completed(futures):
                name, value = fut.result()
                results[name] = value

        wall_ms = (time.perf_counter() - wall_start) * 1000
        self.record(
            f"parallel:{group_name}",
            wall_ms,
            mode="parallel",
            parallel_group=group_name,
            tasks=list(tasks.keys()),
            task_count=len(tasks),
        )
        return results

    def _run_parallel_task(
        self,
        group_name: str,
        name: str,
        fn: Callable[[], T],
    ) -> tuple[str, T]:
        start = time.perf_counter()
        try:
            return name, fn()
        finally:
            self.record(
                name,
                (time.perf_counter() - start) * 1000,
                mode="parallel",
                parallel_group=group_name,
            )

    def _flow_summary(self) -> str:
        """Human-readable ordered flow with parallel groups collapsed."""
        if not self.stages:
            return "(empty)"

        parts: list[str] = []
        seen_groups: set[str] = set()
        for s in self.stages:
            if s.name.startswith("parallel:"):
                parts.append(f"{s.name} [wall {s.duration_ms:.0f}ms]")
                continue
            if s.parallel_group:
                if s.parallel_group not in seen_groups:
                    seen_groups.add(s.parallel_group)
                    tasks = [
                        x.name for x in self.stages
                        if x.parallel_group == s.parallel_group and not x.name.startswith("parallel:")
                    ]
                    parts.append(f"parallel:{s.parallel_group}({', '.join(tasks)})")
            else:
                parts.append(s.name)

        return " → ".join(parts)

    def log_summary(self, *, intent: str = "") -> None:
        total = self.total_ms
        llm_ms = sum(s.duration_ms for s in self.stages if s.name.startswith("llm."))
        ddb_ms = sum(
            s.duration_ms for s in self.stages
            if s.name.startswith("ddb.") or s.name.startswith("parallel:handler.ddb")
        )
        kb_ms = sum(s.duration_ms for s in self.stages if s.name.startswith("classifier.kb"))
        local_ms = max(0.0, total - llm_ms - ddb_ms)

        stage_lines = []
        for s in self.stages:
            pg = f" group={s.parallel_group}" if s.parallel_group else ""
            stage_lines.append(
                f"  #{s.order} {s.name}: {s.duration_ms:.0f}ms [{s.mode}]{pg}"
            )

        logger.info(
            "pipeline_timing",
            extra={
                "request_id": self.request_id,
                "session_id": self.session_id,
                "intent": intent,
                "total_ms": round(total, 2),
                "llm_ms": round(llm_ms, 2),
                "ddb_ms": round(ddb_ms, 2),
                "kb_ms": round(kb_ms, 2),
                "local_ms": round(local_ms, 2),
                "flow": self._flow_summary(),
                "stages": [
                    {
                        "order": s.order,
                        "name": s.name,
                        "ms": s.duration_ms,
                        "mode": s.mode,
                        "parallel_group": s.parallel_group,
                        **s.extra,
                    }
                    for s in self.stages
                ],
            },
        )
        logger.info(
            "pipeline_timing_detail request_id=%s total=%.0fms llm=%.0fms ddb=%.0fms kb=%.0fms local=%.0fms\nflow: %s\n%s",
            self.request_id,
            total,
            llm_ms,
            ddb_ms,
            kb_ms,
            local_ms,
            self._flow_summary(),
            "\n".join(stage_lines),
        )


@contextmanager
def request_timing(
    *,
    request_id: str = "",
    session_id: str = "",
) -> Iterator[PipelineTiming]:
    timing = PipelineTiming(request_id=request_id, session_id=session_id)
    token = _current.set(timing)
    try:
        yield timing
    finally:
        _current.reset(token)


def current() -> PipelineTiming | None:
    return _current.get()


@contextmanager
def stage(
    name: str,
    *,
    mode: str = "sequential",
    parallel_group: str | None = None,
    **extra: Any,
) -> Iterator[None]:
    """Record a stage when a request timing context is active; no-op otherwise."""
    timing = _current.get()
    if timing is None:
        yield
        return
    with timing.stage(name, mode=mode, parallel_group=parallel_group, **extra):
        yield


def record_llm(stage_name: str, duration_ms: float, *, model_id: str = "", **extra: Any) -> None:
    timing = _current.get()
    if timing is None:
        return
    timing.record(
        f"llm.{stage_name}",
        duration_ms,
        mode="sequential",
        model_id=model_id,
        **extra,
    )
