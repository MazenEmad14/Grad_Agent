import base64
import re
from copy import deepcopy
from typing import Any

from state import HematologyGraphState


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _is_valid_base64(value: str) -> bool:
    """Check if string is valid base64-encoded data."""
    try:
        if "," in value:
            value = value.split(",", 1)[1]
        base64.b64decode(value, validate=True)
        return True
    except Exception:
        return False


def _is_valid_url(value: str) -> bool:
    """Minimal URL check — http/https only."""
    pattern = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
    return bool(pattern.match(value))


def _validate_image_field(
    field_name: str,
    value: Any,
    errors: list,
    warnings: list,
) -> bool:
    if value is None:
        return False

    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name}: must be a non-empty string (base64 or URL)")
        return False

    if value.startswith("http://") or value.startswith("https://"):
        if not _is_valid_url(value):
            errors.append(f"{field_name}: invalid URL format")
            return False
        return True

    if not _is_valid_base64(value):
        errors.append(f"{field_name}: invalid base64 encoding")
        return False

    raw = value.split(",", 1)[-1]
    if len(raw) < 100:
        warnings.append(f"{field_name}: base64 content is very short — possible placeholder")

    return True


def _validate_manual_lab_data(
    data: Any,
    errors: list,
    warnings: list,
) -> bool:
    KNOWN_KEYS = {
        "HGB", "RBC", "WBC", "PLT", "MCV", "MCH", "MCHC", 
        "HCT", "NEUT_ABS", "LYMP_ABS", "EOS_ABS", "BASO_ABS", 
        "MONO_ABS", "RDW", "MPV"
    }

    if data is None:
        return False

    if not isinstance(data, dict):
        errors.append("manual_lab_data: must be a dict[str, float]")
        return False

    if len(data) == 0:
        errors.append("manual_lab_data: empty dict provided")
        return False

    for key, val in data.items():
        if not isinstance(key, str):
            errors.append(f"manual_lab_data: key '{key}' must be a string")
            continue
        if not isinstance(val, (int, float)):
            errors.append(f"manual_lab_data['{key}']: value must be numeric, got {type(val).__name__}")
        if key not in KNOWN_KEYS:
            warnings.append(f"manual_lab_data: unrecognized key '{key}' — will be passed through")

    return not any("manual_lab_data[" in e for e in errors)


def _detect_input_type(has_smear: bool, has_lab_image: bool, has_manual: bool) -> str:
    has_image = has_smear or has_lab_image

    if has_image and has_manual:
        return "both"
    if has_image:
        return "image_only"
    if has_manual:
        return "data_only"
    return "unknown"


# ──────────────────────────────────────────────
# Node Class
# ──────────────────────────────────────────────

class ValidateInputsNode:
    """
    Node 1: Validates all inputs and initializes every key
    that downstream nodes will read. Returns dict slice for LangGraph.
    """

    def __call__(self, state: HematologyGraphState) -> dict:
        inp   = state["input_state"]
        agent = deepcopy(state["agent_state"])  # ✅ never mutate original

        # ── 1. Initialize mutable collections ──────────────────────────
        errors:             list = []
        warnings:           list = []
        validation_errors:  list = []
        missing_modalities: list = []
        rule_flags:         list = []
        critical_flags:     list = []

        # ── 2. Validate each raw input ──────────────────────────────────
        has_smear = _validate_image_field(
            "blood_smear_image", inp.get("blood_smear_image"), errors, warnings
        )
        has_lab_image = _validate_image_field(
            "lab_report_image", inp.get("lab_report_image"), errors, warnings
        )
        has_manual = _validate_manual_lab_data(
            inp.get("manual_lab_data"), errors, warnings
        )

        validation_errors.extend(errors)

        # ── 3. Detect / override input_type ────────────────────────────
        detected_type = _detect_input_type(has_smear, has_lab_image, has_manual)
        declared_type = inp.get("input_type", "unknown")

        if declared_type != detected_type:
            warnings.append(
                f"input_type mismatch: declared='{declared_type}', "
                f"detected='{detected_type}' — using detected value"
            )

        # ── 4. Missing modalities ───────────────────────────────────────
        if not has_smear:
            missing_modalities.append("blood_smear_image")
        if not has_lab_image and not has_manual:
            missing_modalities.append("lab_data")

        if detected_type in ("image_only", "data_only"):
            warnings.append(
                f"SINGLE_MODALITY: only '{detected_type}' available — "
                "diagnosis confidence may be reduced"
            )

        # ── 5. is_valid_input ───────────────────────────────────────────
        is_valid = detected_type != "unknown"
        if not is_valid:
            validation_errors.append(
                "NO_USABLE_INPUT: no valid blood_smear_image, "
                "lab_report_image, or manual_lab_data provided"
            )

        # ── 6. Build full agent_state with ALL keys initialized ─────────
        agent.update({
            # Control flow
            "current_node":   "validate_inputs_node",
            "visited_nodes":  ["validate_inputs_node"],
            "decision_trace": ["validate_inputs_node"],

            # Validation results
            "is_valid_input":    is_valid,
            "validation_errors": validation_errors,
            "errors":            errors,
            "warnings":          warnings,

            # Modality tracking
            "missing_modalities": missing_modalities,

            # Safe-init all downstream collections
            "patient_history":   agent.get("patient_history")   or [],
            "trend_timestamps":  agent.get("trend_timestamps")  or [],
            "rule_flags":        rule_flags,
            "critical_flags":    critical_flags,
            "disease_candidates": [],
            "cleaned_data":      {},
            "standardized_data": {},
            "recommendations":   {},
            "explanations":      {},
            "reference_ranges":  agent.get("reference_ranges")  or {},
            "model_versions":    agent.get("model_versions")    or {},

            # Flags — safe defaults
            "is_sick":                    None,
            "disease_type":               None,
            "disease_confidence":         0.0,
            "disease_reasoning":          None,   # ✅ initialized for Node 4 & 6
            "disease_type_confidence":    None,   # ✅ initialized for Node 4
            "severity_level":             None,
            "severity_confidence":        None,   # ✅ initialized for Node 5
            "risk_level":                 None,
            "modality_conflict":          False,
            "low_confidence_flag":        False,
            "requires_doctor":            False,
            "needs_additional_testing":   False,
            "fallback_used":              False,
            "retry_count":                agent.get("retry_count") or 0,
            "data_completeness":          0.0,
            "extracted_text":             None,
            "fusion_insight":             None,
            "fusion_insight_confidence":  None,
            "final_diagnosis":            None,
            "confidence_level":           None,
            "final_recommendations":      None,
            "final_report":               None,
            "trend_analysis":             None,

            # ✅ initialized for Node 6
            "urgent_action_required":  False,
            "doctor_summary":          None,
            "recommendations_title":   None,
            "recommendations_list":    [],
        })

        # ✅ Return BOTH updated agent_state and input_state
        return {
            "agent_state": agent,
            "input_state": {**inp, "input_type": detected_type}
        }


# ──────────────────────────────────────────────
# Ready-to-use instance for LangGraph
# ──────────────────────────────────────────────
validate_inputs_node = ValidateInputsNode()