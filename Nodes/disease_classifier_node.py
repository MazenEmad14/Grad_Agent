import joblib
import pandas as pd
from copy import deepcopy

from llm import get_response
from state import HematologyGraphState

# ─────────────────────────────────────
# Load model bundle
# ─────────────────────────────────────
bundle = joblib.load("models/rf_simple_bundle.joblib")

model             = bundle["model"]
encoder           = bundle["encoder"]
selected_features = bundle["features"]

# ─────────────────────────────────────
# Default fallback values (Updated to match Node 3)
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
class DiseaseClassifierNode:
    """
    Node 4: Classifies the specific disease type for sick patients.
    Skipped automatically when is_sick is False.
    """

    def __call__(self, state: HematologyGraphState) -> dict:
        agent  = deepcopy(state["agent_state"])  # ✅ never mutate original
        errors = list(agent.get("errors") or [])

        # ── Skip if patient is healthy ──────────────────────────────────
        if not agent.get("is_sick", False):
            agent["current_node"]   = "disease_classifier_node"
            agent["visited_nodes"]  = list(agent.get("visited_nodes") or []) + ["disease_classifier_node"]
            agent["decision_trace"] = list(agent.get("decision_trace") or []) + ["disease_classifier_node: Skipped (Patient is healthy)"]
            return {"agent_state": agent}

        data = agent.get("standardized_data", {})

        if not data:
            errors.append("No standardized data available for disease classification.")
            agent["errors"] = errors
            return {"agent_state": agent}

        # ── Build DataFrame input ───────────────────────────────────────
        row = {}
        for feat in selected_features:
            value    = data.get(feat)
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

        # ── LLM Medical Reasoning ───────────────────────────────────────
        prompt = f"""
You are an expert hematologist AI.
The patient has been diagnosed with: {pred} (Confidence: {confidence:.2f}).
Patient's relevant lab values: {data}

TASK:
Write a concise, 2-3 sentence CLINICAL REASONING explaining EXACTLY WHICH specific lab values
(e.g., high WBC, low MCV) led to this specific diagnosis.
Do not write a full report. Do not give advice. Just the medical reasoning.
"""
        # ✅ Try-Except لحماية السيرفر لو الـ LLM وقع
        try:
            reasoning = get_response(prompt)
        except Exception as e:
            reasoning = "Failed to generate LLM reasoning due to API error."
            errors.append(f"LLM Error in Disease Classifier: {str(e)}")

        # ── Update state ────────────────────────────────────────────────
        agent["disease_type"]            = str(pred)
        agent["disease_type_confidence"] = float(confidence)
        agent["disease_reasoning"]       = reasoning
        agent["current_node"]            = "disease_classifier_node"
        agent["visited_nodes"]           = list(agent.get("visited_nodes") or []) + ["disease_classifier_node"]
        agent["decision_trace"]          = list(agent.get("decision_trace") or []) + [f"Disease Classified: {pred} ({confidence:.2f})"]
        agent["errors"]                  = errors

        return {"agent_state": agent}

# ──────────────────────────────────────────────
# Ready-to-use instance for LangGraph
# ──────────────────────────────────────────────
disease_classifier_node = DiseaseClassifierNode()