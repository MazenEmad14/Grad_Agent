import joblib
import pandas as pd
from copy import deepcopy

from state import HematologyGraphState

# ─────────────────────────────────────
# Load severity model
# ─────────────────────────────────────
bundle = joblib.load("models/severity_ai_agent.joblib")

model             = bundle["pipeline"]
encoder           = bundle.get("encoder")
selected_features = bundle["features"]

# ─────────────────────────────────────
# Default fallback values
# ✅ Updated to match Node 3 and Node 4 exactly!
# ─────────────────────────────────────
DEFAULT_VALUES = {
    "HGB": 13.5, "WBC": 7.0, "RBC": 4.7, "HCT": 40.0,
    "MCV": 85.0, "MCH": 29.0, "MCHC": 33.0, "PLT": 250.0,
    "RDW": 13.0, "NEUT_ABS": 4.0, "LYMP_ABS": 2.0, "MONO_ABS": 0.5,
    "EOS_ABS": 0.1, "BASO_ABS": 0.05, "MPV": 9.5,
}

# ─────────────────────────────────────
# Node Class
# ─────────────────────────────────────
class SeverityAssessorNode:
    """
    Node 5: Assesses severity level for sick patients.
    Skipped automatically when is_sick is False.
    """

    def __call__(self, state: HematologyGraphState) -> dict:
        agent  = deepcopy(state["agent_state"])  # ✅ never mutate original
        errors = list(agent.get("errors") or [])

        # ── Skip if patient is healthy ──────────────────────────────────
        # ✅ default=False so is_sick=None does NOT accidentally skip
        if not agent.get("is_sick", False):
            agent["current_node"]   = "severity_assessor_node"
            agent["visited_nodes"]  = list(agent.get("visited_nodes") or []) + ["severity_assessor_node"]  # ✅ + not .append()
            agent["decision_trace"] = list(agent.get("decision_trace") or []) + ["severity_assessor_node: Skipped (Healthy)"]
            agent["severity_level"] = "None"
            return {"agent_state": agent}  # ✅ return dict slice

        data = agent.get("standardized_data", {})

        if not data:
            errors.append("No standardized data for severity assessment.")
            agent["errors"] = errors
            return {"agent_state": agent}

        # ── Build model input ───────────────────────────────────────────
        row = {}
        for feat in selected_features:
            value     = data.get(feat)
            row[feat] = value if value is not None else DEFAULT_VALUES.get(feat, 0)

        input_df = pd.DataFrame([row])

        # ── ML Prediction ───────────────────────────────────────────────
        pred       = model.predict(input_df)[0]
        confidence = 0.0

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(input_df)[0])

        if encoder:
            try:
                pred = encoder.inverse_transform([pred])[0]
            except Exception:
                pass

        # ── Update state ────────────────────────────────────────────────
        agent["severity_level"]      = str(pred)
        agent["severity_confidence"] = float(confidence)
        agent["current_node"]        = "severity_assessor_node"
        agent["visited_nodes"]       = list(agent.get("visited_nodes") or []) + ["severity_assessor_node"]  # ✅
        agent["decision_trace"]      = list(agent.get("decision_trace") or []) + [f"Severity Assessed: {pred} ({confidence:.2f})"]
        agent["errors"]              = errors

        return {"agent_state": agent}  # ✅ return dict slice

# ──────────────────────────────────────────────
# Ready-to-use instance for LangGraph
# ──────────────────────────────────────────────
severity_assessor_node = SeverityAssessorNode()