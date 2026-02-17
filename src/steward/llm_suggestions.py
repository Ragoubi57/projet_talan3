"""LLM Semantic Suggestions – offline-safe stub interface."""
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("nyc_taxi")


class LLMSuggestionEngine:
    """Pluggable LLM interface for data quality suggestions.

    In production, connect to an LLM API. This stub generates
    rule-based suggestions without network access.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("llm", {}).get("enabled", False)
        self.auto_apply = config.get("llm", {}).get("auto_apply_llm_suggestions", False)
        self.suggestions = []

    def generate_suggestions(self, quality_results: list[dict]) -> list[dict]:
        """Generate suggestions based on failed quality checks."""
        self.suggestions = []

        for result in quality_results:
            if result["passed"]:
                continue

            suggestion = self._make_suggestion(result)
            if suggestion:
                self.suggestions.append(suggestion)

        logger.info(f"LLM stub generated {len(self.suggestions)} suggestions")
        return self.suggestions

    def _make_suggestion(self, check_result: dict) -> dict | None:
        """Generate a suggestion for a failed check (rule-based stub)."""
        check = check_result["check"]
        fail_pct = check_result["fail_pct"]

        suggestion_map = {
            "temporal_sanity": {
                "rule": "Swap pickup and dropoff datetimes when pickup >= dropoff",
                "rationale": "Likely data entry error; timestamps are reversed",
                "action": "swap_datetime",
                "priority": "high",
            },
            "passenger_count_range": {
                "rule": "Clamp passenger_count to [1, 6] and impute nulls with median",
                "rationale": "Values outside range indicate sensor/entry errors",
                "action": "clamp_passenger",
                "priority": "medium",
            },
            "trip_distance_non_negative": {
                "rule": "Take absolute value of negative trip distances",
                "rationale": "Negative distance is physically impossible",
                "action": "abs_distance",
                "priority": "high",
            },
            "fare_non_negative": {
                "rule": "Winsorize fare outliers and fix negative values",
                "rationale": "Negative fares suggest refunds or errors",
                "action": "winsorize_fare",
                "priority": "high",
            },
            "zero_duration_positive_distance": {
                "rule": "Drop trips with zero duration but positive distance",
                "rationale": "Physically impossible; likely GPS/meter error",
                "action": "drop_zero_duration",
                "priority": "medium",
            },
        }

        for key, suggestion in suggestion_map.items():
            if key in check:
                return {
                    "check": check,
                    "fail_pct": fail_pct,
                    "suggested_at": datetime.now().isoformat(),
                    "auto_apply": self.auto_apply,
                    **suggestion,
                }

        # Generic suggestion for other failures
        if fail_pct > 1.0:
            return {
                "check": check,
                "fail_pct": fail_pct,
                "suggested_at": datetime.now().isoformat(),
                "rule": f"Investigate {check} failures ({fail_pct}% affected)",
                "rationale": "Significant number of rows failing this check",
                "action": "manual_review",
                "priority": "low",
                "auto_apply": False,
            }
        return None

    def save_suggestions(self, output_json: str, output_md: str):
        """Save suggestions to JSON review queue and Markdown report."""
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)

        with open(output_json, "w") as f:
            json.dump(self.suggestions, f, indent=2)

        md_lines = [
            "# LLM Semantic Suggestions",
            f"\n**Generated:** {datetime.now().isoformat()}",
            f"**Provider:** stub (offline-safe)",
            f"**Auto-apply:** {self.auto_apply}",
            f"\n## Suggestions ({len(self.suggestions)} total)\n",
        ]
        for i, s in enumerate(self.suggestions, 1):
            md_lines.extend([
                f"### {i}. {s['check']}",
                f"- **Rule:** {s['rule']}",
                f"- **Rationale:** {s['rationale']}",
                f"- **Action:** {s['action']}",
                f"- **Priority:** {s['priority']}",
                f"- **Fail %:** {s['fail_pct']}%",
                f"- **Auto-apply:** {s.get('auto_apply', False)}",
                "",
            ])

        with open(output_md, "w") as f:
            f.write("\n".join(md_lines))

        logger.info(f"LLM suggestions saved: {output_json}, {output_md}")
