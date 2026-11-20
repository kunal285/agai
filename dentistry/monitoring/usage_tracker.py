"""OpenAI API token usage and cost monitoring."""
from __future__ import annotations
from typing import Any
import threading

# Estimated pricing per 1M tokens (USD)
MODEL_PRICING = {
    # Primary model
    "gpt-6.1-sol": {"prompt": 2.50, "completion": 10.00},
    # Advanced model
    "gpt-6.2-astra": {"prompt": 5.00, "completion": 20.00},
    # Budget model
    "gpt-6-luna": {"prompt": 0.50, "completion": 2.00},
    # Fallback / standard defaults
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
}

class UsageTracker:
    """Thread-safe tracker for API calls, token counts, latency, and estimated cost."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_latency_ms: float = 0.0
        self.errors: int = 0
        self.history: list[dict[str, Any]] = []

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        agent_name: str = "orchestrator",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            tot = prompt_tokens + completion_tokens
            self.total_tokens += tot
            self.total_latency_ms += latency_ms

            pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-6.1-sol"])
            cost = (prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]) / 1_000_000.0

            record = {
                "call_id": self.calls,
                "agent": agent_name,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": tot,
                "latency_ms": round(latency_ms, 2),
                "estimated_cost_usd": round(cost, 6),
                "metadata": metadata or {},
            }
            self.history.append(record)
            return record

    def record_error(self, agent_name: str = "orchestrator", error: str = "") -> None:
        with self._lock:
            self.errors += 1
            self.history.append({
                "call_id": self.calls + 1,
                "agent": agent_name,
                "status": "error",
                "error": error,
            })

    def get_summary(self, default_model: str = "gpt-6.1-sol") -> dict[str, Any]:
        with self._lock:
            pricing = MODEL_PRICING.get(default_model, MODEL_PRICING["gpt-6.1-sol"])
            total_cost = (
                self.prompt_tokens * pricing["prompt"] + self.completion_tokens * pricing["completion"]
            ) / 1_000_000.0

            return {
                "api_calls": self.calls,
                "errors": self.errors,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "avg_latency_ms": round(self.total_latency_ms / max(1, self.calls), 2),
                "total_estimated_cost_usd": round(total_cost, 6),
                "model_pricing_reference": pricing,
            }
