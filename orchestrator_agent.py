"""
╔══════════════════════════════════════════════════════════════╗
║              ORCHESTRATOR AGENT — MULTI-AGENT SYSTEM         ║
║                                                              ║
║  Coordinates the entire pipeline:                            ║
║  1. Is there drift?        → DriftAgent                      ║
║  2. Is the model degraded? → AlertAgent                      ║
║  3. Is retraining needed?  → RetrainAgent                    ║
║  4. What to tell the exec? → ReportAgent                     ║
║                                                              ║
║  Each agent makes its own decision; orchestrator combines.   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import glob
import sqlite3
import subprocess
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

# ── CONFIG ────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME    = "llama-3.3-70b-versatile"
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
REPORTS_DIR = "reports"
LOGS_DIR    = "logs"
MLFLOW_DB   = "mlflow.db"
LOG_PATH    = f"{LOGS_DIR}/orchestrator.log"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# TOOLS — Shared across all agents
# ══════════════════════════════════════════════════════════════

def tool_get_drift_status() -> dict:
    """Reads the most recent drift report."""
    reports = glob.glob(f"{REPORTS_DIR}/drift_report_*.json")
    if not reports:
        return {"drift_detected": False, "reason": "No drift report found yet"}
    latest = max(reports, key=os.path.getctime)
    with open(latest) as f:
        data = json.load(f)
    return {
        "drift_detected" : data.get("drift_detected", False),
        "sample_size"    : data.get("sample_size", 0),
        "timestamp"      : data.get("timestamp", "unknown"),
        "features"       : data.get("features", []),
        "report_file"    : latest
    }


def tool_get_model_accuracy() -> dict:
    """Fetches accuracy history from the last 5 MLflow runs (ordered by start_time)."""
    try:
        conn = sqlite3.connect(MLFLOW_DB)
        cur  = conn.cursor()
        cur.execute("""
            SELECT m.value, r.start_time
            FROM metrics m
            JOIN runs r ON m.run_uuid = r.run_uuid
            WHERE m.key = 'accuracy'
            ORDER BY r.start_time DESC LIMIT 5
        """)
        rows = cur.fetchall()
        conn.close()
        values = [r[0] for r in rows]
        return {
            "recent_accuracies" : values,
            "latest_accuracy"   : values[0] if values else None,
            "trend"             : "declining"  if len(values) >= 2 and values[0] < values[1]
                                  else "improving" if len(values) >= 2 and values[0] > values[1]
                                  else "stable"
        }
    except Exception as e:
        return {"error": str(e), "latest_accuracy": None}


def tool_get_return_rate() -> dict:
    """Reads the return rate from the most recent pipeline report."""
    reports = glob.glob(f"{REPORTS_DIR}/report_*.txt")
    if not reports:
        return {"return_rate": None, "reason": "No reports found"}
    latest = max(reports, key=os.path.getctime)
    with open(latest, encoding="utf-8") as f:
        for line in f:
            if "return_rate" in line:
                try:
                    rate = float(line.split(":")[1].strip())
                    return {"return_rate": rate, "source_file": latest}
                except Exception:
                    pass
    return {"return_rate": None, "reason": "return_rate line not found"}


def tool_get_alert_history() -> dict:
    """Reads the last 5 alert log entries."""
    log_file = f"{LOGS_DIR}/alerts.log"
    if not os.path.exists(log_file):
        return {"alerts": [], "degradation_count": 0}
    with open(log_file, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    blocks = [b.strip() for b in content.split("=" * 60) if b.strip()][-5:]
    degradation_count = sum(1 for b in blocks if "MODEL_DEGRADATION" in b)
    return {
        "recent_alerts"     : blocks,
        "degradation_count" : degradation_count,
        "total_reviewed"    : len(blocks)
    }


def tool_run_pipeline() -> dict:
    """Executes main_pipeline.py to retrain the model."""
    print("    [RetrainAgent] Running main_pipeline.py ...")
    # Use sys.executable (the same Python that is running this script,
    # i.e. the venv's interpreter) instead of the bare "python" command.
    # This avoids accidentally invoking a different, unmigrated global
    # Python/mlflow installation from PATH.
    #
    # Also force UTF-8 I/O in the child process via PYTHONIOENCODING.
    # Without this, Windows' default console codepage (cp1252) can't
    # encode characters like "✓" that main_pipeline.py prints, which
    # crashes the child process with a UnicodeEncodeError.
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "main_pipeline.py"],
        capture_output=True, text=True, encoding="utf-8", env=child_env
    )
    success = result.returncode == 0

    if not success:
        # Print the real error to the console so failures are visible
        # immediately, instead of only being buried in the log file.
        print("    ┌─ RETRAIN ERROR ─────────────────────────────────")
        print(f"    │ Exit code: {result.returncode}")
        error_text = (result.stderr or "").strip()
        if error_text:
            for line in error_text.splitlines()[-15:]:
                print(f"    │ {line}")
        else:
            print("    │ (no stderr output captured)")
        print("    └──────────────────────────────────────────────────")

    return {
        "success"   : success,
        "output"    : result.stdout[-500:] if success else result.stderr[-1500:],
        "timestamp" : datetime.now().isoformat()
    }


def tool_get_latest_report_summary() -> dict:
    """Returns the first 800 characters of the latest pipeline report."""
    reports = glob.glob(f"{REPORTS_DIR}/report_*.txt")
    if not reports:
        return {"summary": "No reports available yet"}
    latest = max(reports, key=os.path.getctime)
    with open(latest, encoding="utf-8") as f:
        content = f.read()
    return {
        "summary"   : content[:800],
        "full_path" : latest,
        "generated" : datetime.fromtimestamp(os.path.getctime(latest)).strftime("%Y-%m-%d %H:%M")
    }


# ══════════════════════════════════════════════════════════════
# HELPER — Ask LLM for a short, reliable response
# ══════════════════════════════════════════════════════════════

def _ask_llm(prompt: str) -> str:
    """
    Sends a prompt to the LLM.
    Retries once with stronger anti-repetition settings if the
    output looks broken (empty, too short, or repetition loops).
    """
    def is_broken(text: str) -> tuple[bool, str]:
        if not text or len(text.strip()) < 10:
            return True, "empty or too short"
        if text.count("@") > 10:
            return True, "too many '@' characters (repetition loop)"
        # Relaxed from < 5 to < 8 unique chars over a reasonable length —
        # a real repetition loop (e.g. "aaaaaaa...") has very few unique
        # characters; a normal multi-sentence report does not trip this.
        if len(text) > 30 and len(set(text)) < 8:
            return True, "too few unique characters (likely stuck in a loop)"
        return False, ""

    attempts = [
        {"temperature": 0.2},
        {"temperature": 0.1},
    ]

    for i, options in enumerate(attempts, start=1):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )
            text = response.choices[0].message.content.strip()
            broken, reason = is_broken(text)
            if not broken:
                return text
            preview = text[:200].replace("\n", " ")
            print(f"    [LLM] Attempt {i} rejected ({reason}). Preview: {preview!r}")
        except Exception as e:
            print(f"    [LLM] Attempt {i} raised an error: {e}")
            return f"[LLM error: {e}]"

    return "LLM could not produce a reliable output for this step."


# ══════════════════════════════════════════════════════════════
# AGENT 1 — DriftAgent
# ══════════════════════════════════════════════════════════════

class DriftAgent:
    """
    Reads the latest drift report and produces a concise diagnosis.
    Decision: drift detected → True / False
    """

    def run(self) -> dict:
        print("\n  [DriftAgent] Running ...")
        drift_data = tool_get_drift_status()

        prompt = f"""You performed a data drift analysis. Results:
Drift detected: {drift_data.get('drift_detected')}
Sample size: {drift_data.get('sample_size')}
Feature details: {json.dumps(drift_data.get('features', []), indent=2)}

Based only on the data above, write a 3-sentence assessment:
1. Is there drift, or is the distribution stable?
2. What is the urgency level?
3. What do you recommend?
Do not invent information not present in the data."""

        diagnosis = _ask_llm(prompt)

        result = {
            "agent"          : "DriftAgent",
            "drift_detected" : drift_data.get("drift_detected", False),
            "diagnosis"      : diagnosis,
            "raw_data"       : drift_data
        }
        print(f"    → Drift: {'DETECTED ⚠️' if result['drift_detected'] else 'NONE ✅'}")
        return result


# ══════════════════════════════════════════════════════════════
# AGENT 2 — AlertAgent
# ══════════════════════════════════════════════════════════════

class AlertAgent:
    """
    Monitors model accuracy and return rate against thresholds.
    Decision: issues found → True / False
    """

    ACCURACY_THRESHOLD = 0.85
    RETURN_RATE_LIMIT  = 25.0

    def run(self) -> dict:
        print("\n  [AlertAgent] Running ...")
        accuracy_data = tool_get_model_accuracy()
        return_data   = tool_get_return_rate()
        alert_history = tool_get_alert_history()

        latest_acc  = accuracy_data.get("latest_accuracy")
        return_rate = return_data.get("return_rate")

        issues = []
        if latest_acc is not None and latest_acc < self.ACCURACY_THRESHOLD:
            issues.append(
                f"Model accuracy low: {latest_acc:.4f} "
                f"(threshold: {self.ACCURACY_THRESHOLD})"
            )
        if return_rate is not None and return_rate > self.RETURN_RATE_LIMIT:
            issues.append(
                f"Return rate high: {return_rate}% "
                f"(threshold: {self.RETURN_RATE_LIMIT}%)"
            )

        prompt = f"""Model monitoring results:
Recent accuracy values: {accuracy_data.get('recent_accuracies')}
Accuracy trend: {accuracy_data.get('trend')}
Return rate: {return_rate}%
MODEL_DEGRADATION alerts in last {alert_history.get('total_reviewed')} checks: {alert_history.get('degradation_count')}
Detected issues: {issues if issues else 'None'}

Write a 3-4 sentence status assessment based strictly on the data above.
Do not add information that is not present."""

        diagnosis = _ask_llm(prompt)

        result = {
            "agent"           : "AlertAgent",
            "has_issues"      : len(issues) > 0,
            "issues"          : issues,
            "latest_accuracy" : latest_acc,
            "return_rate"     : return_rate,
            "diagnosis"       : diagnosis
        }
        status = "ISSUES FOUND ⚠️" if result["has_issues"] else "ALL CLEAR ✅"
        print(f"    → Status: {status}")
        for issue in issues:
            print(f"      • {issue}")
        return result


# ══════════════════════════════════════════════════════════════
# AGENT 3 — RetrainAgent
# ══════════════════════════════════════════════════════════════

class RetrainAgent:
    """
    Decides whether retraining is needed based on DriftAgent
    and AlertAgent results, then executes it if required.
    """

    def run(self, drift_result: dict, alert_result: dict) -> dict:
        print("\n  [RetrainAgent] Running ...")

        should_retrain = drift_result["drift_detected"] or alert_result["has_issues"]

        prompt = f"""Retraining decision evaluation:
Drift detected: {drift_result['drift_detected']}
Model issues found: {alert_result['has_issues']}
Detected issues: {alert_result.get('issues', [])}
Latest accuracy: {alert_result.get('latest_accuracy')}

Is the retraining decision justified?
Write 2 sentences of reasoning based only on the data above."""

        reasoning = _ask_llm(prompt)

        retrain_output = None
        if should_retrain:
            print("    → Triggering retraining ... ⚙️")
            retrain_output = tool_run_pipeline()
            success = retrain_output.get("success", False)
            print(f"    → Retrain: {'COMPLETE ✅' if success else 'FAILED ❌'}")
        else:
            print("    → No retraining needed ✅")

        return {
            "agent"          : "RetrainAgent",
            "should_retrain" : should_retrain,
            "reasoning"      : reasoning,
            "retrain_done"   : should_retrain,
            "retrain_output" : retrain_output
        }


# ══════════════════════════════════════════════════════════════
# AGENT 4 — ReportAgent
# ══════════════════════════════════════════════════════════════

class ReportAgent:
    """
    Combines all agent outputs into a concise executive summary.
    """

    def run(self, drift: dict, alert: dict, retrain: dict) -> dict:
        print("\n  [ReportAgent] Running ...")

        report_summary = tool_get_latest_report_summary()

        prompt = f"""Daily MLOps pipeline run summary:

DRIFT STATUS:
{drift['diagnosis']}

MODEL / ALERT STATUS:
{alert['diagnosis']}

RETRAIN DECISION:
{retrain['reasoning']}
Retraining executed: {'Yes' if retrain['retrain_done'] else 'No'}

LATEST REPORT EXCERPT:
{report_summary.get('summary', 'Not available')}

Write a concise executive summary in English (5-6 sentences):
1. What is the overall system status?
2. What actions were taken today?
3. Is there anything that needs attention?
Base your answer strictly on the data provided. Do not exaggerate."""

        executive_summary = _ask_llm(prompt)

        result = {
            "agent"             : "ReportAgent",
            "executive_summary" : executive_summary,
            "report_source"     : report_summary.get("full_path"),
            "generated_at"      : report_summary.get("generated")
        }
        print("    → Executive summary generated ✅")
        return result


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR — Coordinates all agents
# ══════════════════════════════════════════════════════════════

class OrchestratorAgent:
    """
    Multi-agent coordinator.

    Execution flow:
    DriftAgent → AlertAgent → RetrainAgent → ReportAgent → Log
    """

    def __init__(self):
        self.drift_agent   = DriftAgent()
        self.alert_agent   = AlertAgent()
        self.retrain_agent = RetrainAgent()
        self.report_agent  = ReportAgent()

    def run(self) -> dict:
        start_time = datetime.now()

        print("\n" + "═" * 60)
        print("  ORCHESTRATOR AGENT — STARTING")
        print(f"  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 60)

        # ── STEP 1: Drift check ────────────────────────────
        print("\n[1/4] Calling DriftAgent ...")
        drift_result = self.drift_agent.run()

        # ── STEP 2: Model & alert check ───────────────────
        print("\n[2/4] Calling AlertAgent ...")
        alert_result = self.alert_agent.run()

        # ── STEP 3: Retrain decision ───────────────────────
        print("\n[3/4] Calling RetrainAgent ...")
        retrain_result = self.retrain_agent.run(drift_result, alert_result)

        # ── STEP 4: Executive report ───────────────────────
        print("\n[4/4] Calling ReportAgent ...")
        report_result = self.report_agent.run(drift_result, alert_result, retrain_result)

        # ── FINAL ──────────────────────────────────────────
        end_time = datetime.now()
        duration = (end_time - start_time).seconds

        final = {
            "orchestrator_run" : {
                "started_at"   : start_time.isoformat(),
                "finished_at"  : end_time.isoformat(),
                "duration_sec" : duration
            },
            "drift"   : drift_result,
            "alert"   : alert_result,
            "retrain" : retrain_result,
            "report"  : report_result
        }

        self._print_summary(final)
        self._save_log(final)

        return final

    def _print_summary(self, result: dict):
        print("\n" + "═" * 60)
        print("  ORCHESTRATOR — SUMMARY")
        print("═" * 60)

        d   = result["drift"]
        a   = result["alert"]
        r   = result["retrain"]
        rep = result["report"]

        print(f"\n  Drift Status    : {'⚠️  DETECTED' if d['drift_detected'] else '✅ NONE'}")
        print(f"  Model Issues    : {'⚠️  YES'       if a['has_issues']      else '✅ NO'}")
        print(f"  Retrain Done    : {'✅ YES'         if r['retrain_done']    else '➖  NO'}")
        print(f"  Duration        : {result['orchestrator_run']['duration_sec']} seconds")

        print("\n" + "─" * 60)
        print("  EXECUTIVE SUMMARY:")
        print("─" * 60)
        print(f"\n{rep['executive_summary']}\n")
        print("═" * 60)

    def _save_log(self, result: dict):
        """Appends the orchestrator run to the log file as JSON."""
        log_entry = {
            "timestamp" : result["orchestrator_run"]["started_at"],
            "drift"     : result["drift"]["drift_detected"],
            "issues"    : result["alert"]["issues"],
            "retrained" : result["retrain"]["retrain_done"],
            "summary"   : result["report"]["executive_summary"]
        }

        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=2))
            f.write("\n")

        print(f"\n  Log saved: {LOG_PATH}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    orchestrator = OrchestratorAgent()
    result = orchestrator.run()