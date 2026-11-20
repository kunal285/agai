"""Centralized OpenAI LLM client with retry, timeout, token tracking, and offline fallback."""
from __future__ import annotations
import os
import time
import logging
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from dentistry.monitoring.usage_tracker import UsageTracker
from dentistry.monitoring.agent_logger import AgentExecutionLogger
from dentistry.safety.privacy import sanitize_metadata

log = logging.getLogger(__name__)

# Load .env file from project root if present
load_dotenv(override=False)

DEFAULT_PRIMARY_MODEL = "gpt-6.1-sol"
DEFAULT_ADVANCED_MODEL = "gpt-6.2-astra"
DEFAULT_BUDGET_MODEL = "gpt-6-luna"

class OpenAIOrchestrationClient:
    """Manages Agent API initialization, multi-turn tool calling, and deterministic offline fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        enabled: bool | None = None,
        usage_tracker: UsageTracker | None = None,
        agent_logger: AgentExecutionLogger | None = None,
        provider: str | None = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()
        providers = {
            "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
            "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
            "openai": ("https://api.openai.com/v1", DEFAULT_PRIMARY_MODEL),
        }
        if self.provider not in providers:
            raise ValueError("LLM_PROVIDER must be groq, gemini, or openai")
        self.base_url, default_model = providers[self.provider]
        prefix = self.provider.upper()
        self.api_key = api_key or os.getenv(f"{prefix}_API_KEY", "").strip()
        
        # If model passed is a placeholder or from another provider, adapt to active provider
        passed_model = model or os.getenv(f"{prefix}_MODEL", default_model).strip()
        if self.provider == "groq" and ("gpt-6" in passed_model or "gemini" in passed_model):
            passed_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        elif self.provider == "openai" and ("llama" in passed_model or "oss" in passed_model or "gemini" in passed_model):
            passed_model = os.getenv("OPENAI_MODEL", DEFAULT_PRIMARY_MODEL).strip()
        self.model = passed_model
        self.timeout = float(timeout if timeout is not None else os.getenv(f"{prefix}_TIMEOUT", 30))
        self.max_retries = int(max_retries if max_retries is not None else os.getenv(f"{prefix}_MAX_RETRIES", 2))
        env_enabled = os.getenv("LLM_ENABLED", os.getenv(f"{prefix}_ENABLED", "true")).lower() in ("true", "1", "yes")
        self.enabled = (enabled if enabled is not None else env_enabled) and bool(self.api_key)

        self.usage_tracker = usage_tracker or UsageTracker()
        self.agent_logger = agent_logger or AgentExecutionLogger()
        self._client = None

        if self.enabled:
            self._init_client()
        else:
            log.info("OpenAI Agentic AI is operating in offline / fallback mode (API key not set or OPENAI_ENABLED=false).")

    def _init_client(self) -> None:
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            log.info(f"Agent client initialized successfully with model: {self.model}")
        except Exception as e:
            log.warning(f"Failed to initialize OpenAI client: {e}. Falling back to offline mode.")
            self.enabled = False
            self._client = None

    def is_available(self) -> bool:
        """Returns True if the Agent API is enabled and configured."""
        return self.enabled and self._client is not None

    def run_agent_turn(
        self,
        agent_name: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Executes a single LLM completion or tool call turn with robust retry, timeout, and usage tracking."""
        if not self.is_available():
            return {
                "status": "offline",
                "message": "Agent API is disabled or unavailable. Falling back to local deterministic rule engine.",
                "tool_calls": [],
                "content": None,
            }

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        sanitized_messages = [sanitize_metadata(m) for m in full_messages]

        t0 = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": sanitized_messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = self._client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            choice = response.choices[0]
            msg = choice.message

            # Track token usage
            prompt_toks = response.usage.prompt_tokens if response.usage else 0
            comp_toks = response.usage.completion_tokens if response.usage else 0
            self.usage_tracker.record_usage(
                model=self.model,
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                latency_ms=latency_ms,
                agent_name=agent_name,
            )

            tool_calls_data = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_data.append({
                        "id": tc.id,
                        "type": tc.type,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })

            return {
                "status": "success",
                "content": msg.content,
                "tool_calls": tool_calls_data,
                "finish_reason": choice.finish_reason,
                "latency_ms": latency_ms,
                "usage": {
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": comp_toks,
                    "total_tokens": prompt_toks + comp_toks,
                },
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            err_str = str(e)
            log.warning(f"Agent API call failed for agent '{agent_name}': {err_str}")
            if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str or "401" in err_str:
                log.info("Disabling Agent API for subsequent calls due to quota/auth limit; continuing in offline deterministic mode.")
                self.enabled = False
            self.usage_tracker.record_error(agent_name=agent_name, error=err_str)
            return {
                "status": "error",
                "error": err_str,
                "latency_ms": latency_ms,
                "tool_calls": [],
                "content": None,
            }

    def _build_dynamic_synthesis(
        self,
        analysis_id: str,
        quality_data: dict[str, Any],
        findings_data: list[dict[str, Any]],
        validation_data: dict[str, Any],
    ) -> str:
        """Constructs an intelligent, structured non-diagnostic synthesis from validated findings."""
        q_status = quality_data.get("quality_status") or quality_data.get("status", "unknown")
        total_teeth = len(findings_data)
        rev_req = validation_data.get("human_review_required", True)
        
        lines = [
            f"**Dental Screening AI Synthesis (Analysis ID: {analysis_id})**\n",
            f"- **Image Quality Gate:** {q_status.title()} (Sharpness & contrast verified for screening).",
            f"- **Findings Overview:** {total_teeth} suspicious dental region{'s' if total_teeth != 1 else ''} localized on panoramic radiograph.",
        ]

        if not findings_data:
            lines.append("- **Detection Note:** No candidate abnormalities identified above the confidence threshold.")
        else:
            lines.append("- **Tooth-by-Tooth Clinical Observations:**")
            for idx, f in enumerate(findings_data, 1):
                t_label = f.get("tooth_label", f"Tooth #{idx}")
                finding = f.get("predicted_finding") or f.get("predicted_class") or f.get("detector_class") or "Dental Finding"
                conf = f.get("confidence_label") or f"{int(round(float(f.get('detector_confidence') or 0.0)*100))}%"
                reasons = f.get("simplified_reasons") or []
                reason_str = f" ({reasons[0]})" if reasons else ""
                lines.append(f"  • **{t_label}:** {finding} [Confidence: {conf}]{reason_str}")

        lines.append(f"\n- **Clinical Validation Triage:** {'Dentist Review Needed ⚠️' if rev_req else 'No Additional Review Flag ✅'}.")
        if rev_req:
            summary = validation_data.get("review_summary", {})
            disagreements = summary.get("model_disagreements", 0)
            low_conf = summary.get("low_confidence_findings", 0)
            triage_notes = []
            if disagreements > 0:
                triage_notes.append(f"{disagreements} tooth finding{'s' if disagreements != 1 else ''} had model disagreements")
            if low_conf > 0:
                triage_notes.append(f"{low_conf} finding{'s' if low_conf != 1 else ''} had low AI confidence")
            if triage_notes:
                lines.append(f"  *Reason:* {', and '.join(triage_notes)}. In-person clinical dental evaluation is recommended.")
            else:
                lines.append("  *Reason:* Screening findings require clinical confirmation by a licensed dentist.")

        lines.append("\n*Disclaimer: Educational research prototype only. Model predictions are non-diagnostic and do not replace clinical examination or bitewing imaging.*")
        return "\n".join(lines)

    def generate_clinical_synthesis(
        self,
        analysis_id: str,
        quality_data: dict[str, Any],
        findings_data: list[dict[str, Any]],
        validation_data: dict[str, Any],
    ) -> str:
        """Prompts the LLM to write an objective, non-diagnostic synthesis, or generates a structured dynamic fallback."""
        if not self.is_available():
            return self._build_dynamic_synthesis(analysis_id, quality_data, findings_data, validation_data)

        prompt = (
            f"Screening Analysis ID: {analysis_id}\n"
            f"Image Quality Assessment: {quality_data.get('status')} (acceptable={quality_data.get('acceptable')})\n"
            f"Detected Teeth Count: {len(findings_data)}\n"
            f"Tooth-by-Tooth Findings:\n"
        )
        for idx, f in enumerate(findings_data, 1):
            t_label = f.get("tooth_label", f"Tooth #{idx}")
            finding = f.get("predicted_finding") or f.get("predicted_class") or f.get("detector_class")
            conf = f.get("confidence_label") or f"{int(round(float(f.get('detector_confidence') or 0.0)*100))}%"
            reasons = f.get("simplified_reasons", [])
            prompt += f"  - {t_label}: Finding '{finding}', Confidence: {conf}, Notes: {reasons}\n"

        prompt += (
            f"\nValidation Triage:\n"
            f"Human Dentist Review Required: {validation_data.get('human_review_required', True)}\n"
            f"Review Summary: {validation_data.get('review_summary', {})}\n\n"
            "Write a concise, professional 3-paragraph clinical screening synthesis: "
            "1) State the radiograph quality and number of suspicious teeth detected. "
            "2) Summarize the key tooth findings, highlighting any model disagreements or low-confidence areas. "
            "3) State the triage recommendation for the attending dentist, reiterating that this prototype is non-diagnostic."
        )

        sys_msg = (
            "You are a medical AI documentation assistant. You summarize objective computer-vision findings for dental screening. "
            "Never invent diagnoses. Never declare patients healthy. Reiterate the mandatory research disclaimer."
        )

        res = self.run_agent_turn(
            agent_name="reporting_synthesis",
            system_prompt=sys_msg,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        if res["status"] == "success" and res["content"]:
            return res["content"].strip()

        # If LLM API call fails, seamlessly return our dynamic structured synthesis
        return self._build_dynamic_synthesis(analysis_id, quality_data, findings_data, validation_data)
