"""Agent execution and lifecycle logging."""
from __future__ import annotations
import time
import logging
from typing import Any
from datetime import datetime, timezone

log = logging.getLogger(__name__)

class AgentExecutionLogger:
    """Maintains a structured in-memory execution trace of all agent actions, timings, and transitions."""

    def __init__(self):
        self.logs: list[dict[str, Any]] = []
        self._active_timers: dict[str, float] = {}

    def log_event(self, agent_name: str, event_type: str, details: dict[str, Any] | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "event_type": event_type,
            "details": details or {},
        }
        self.logs.append(entry)
        log.info(f"[{agent_name}] {event_type}: {details}")

    def start_agent(self, agent_name: str) -> None:
        self._active_timers[agent_name] = time.perf_counter()
        self.log_event(agent_name, "started")

    def complete_agent(self, agent_name: str, status: str = "success", extra: dict[str, Any] | None = None) -> float:
        t0 = self._active_timers.pop(agent_name, None)
        duration_ms = (time.perf_counter() - t0) * 1000.0 if t0 else 0.0
        payload = {"status": status, "duration_ms": round(duration_ms, 2)}
        if extra:
            payload.update(extra)
        self.log_event(agent_name, "completed", payload)
        return duration_ms

    def fail_agent(self, agent_name: str, error_msg: str) -> float:
        t0 = self._active_timers.pop(agent_name, None)
        duration_ms = (time.perf_counter() - t0) * 1000.0 if t0 else 0.0
        self.log_event(agent_name, "failed", {"error": error_msg, "duration_ms": round(duration_ms, 2)})
        return duration_ms

    def get_summary(self) -> dict[str, Any]:
        durations: dict[str, float] = {}
        statuses: dict[str, str] = {}
        for entry in self.logs:
            if entry["event_type"] in ("completed", "failed"):
                agent = entry["agent"]
                durations[agent] = entry["details"].get("duration_ms", 0.0)
                statuses[agent] = entry["details"].get("status", "unknown")

        return {
            "total_events": len(self.logs),
            "agent_statuses": statuses,
            "agent_latencies_ms": durations,
            "total_latency_ms": round(sum(durations.values()), 2),
            "trace": list(self.logs),
        }
