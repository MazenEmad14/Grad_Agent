"""
Node 2: ExtractAndStandardizeNode
──────────────────────────────────────
المسؤولية (LLM Driven 100%):
  - استخراج النص من الصورة (OCR).
  - تمرير كل القواميس للـ LLM في البرومبت.
  - الـ LLM يولد قاموس شامل يحتوي على "كل" المسميات (القديمة، القصيرة، والطويلة بالوحدات).
  - لا يوجد Status.
"""

import json
import re
from copy import deepcopy

from state import HematologyGraphState
from llm import get_response, get_vision_response


# ──────────────────────────────────────────────
# Dictionaries to inject into the Prompt
# ──────────────────────────────────────────────

KEY_ALIASES = {
    "HGB":  ["hemoglobin", "hgb", "hb", "haemoglobin", "hemoglobin (g/dl)"],
    "WBC":  ["white blood cells", "wbc", "leukocytes", "wbc (cells/µl)"],
    "RBC":  ["red blood cells", "rbc", "erythrocytes", "rbc (million/µl)"],
    "HCT":  ["hematocrit", "hct", "packed cell volume", "pcv", "hematocrit (%)"],
    "MCV":  ["mean corpuscular volume", "mcv", "mcv (fl)"],
    "MCH":  ["mean corpuscular hemoglobin", "mch", "mch (pg)"],
    "MCHC": ["mean corpuscular hemoglobin concentration", "mchc", "mchc (g/dl)"],
    "PLT":  ["plt", "platelets", "thrombocytes", "platelet count", "platelet count (cells/µl)", "platelets count"],
    "RDW":  ["rdw", "rdw-cv", "red cell distribution width", "rdw (%)"],
    "NEUT": ["neu", "neutrophils", "neut", "neutrophils (%)", "neutrophils (absolute)", "neut_abs"],
    "LYMP": ["lym", "lymphocytes", "lymph", "lymphocytes (%)", "lymphocytes (absolute)", "lymp_abs"],
    "MONO": ["mon", "monocytes", "mono", "monocytes (%)", "monocytes (absolute)", "mono_abs"],
    "EOS":  ["eos", "eosinophils", "eosinophil", "eosinophils (%)", "eos_abs"],
    "BASO": ["bas", "basophils", "basophil", "basophils (%)", "baso_abs"],
    "MPV":  ["mpv", "mean platelet volume"],
}

TARGET_COLUMNS = [
    "HGB", "Hb", "WBC", "RBC", "HCT", "MCV", "MCH", "MCHC",
    "PLT", "RDW", "NEUT", "NEUT_ABS", "LYMP", "LYMP_ABS",
    "MONO", "MONO_ABS", "EOS", "EOS_ABS", "BASO", "BASO_ABS", "MPV",
]

NORMAL_VALUES = {
    "HGB": 14.0, "Hb": 14.0, "WBC": 7.0,  "RBC": 4.8,  "HCT": 42.0, "MCV": 85.0,
    "MCH": 29.0, "MCHC": 33.0, "PLT": 250.0, "RDW": 13.0,
    "NEUT": 55.0, "NEUT_ABS": 4.0, "LYMP": 30.0, "LYMP_ABS": 2.5,
    "MONO": 5.0,  "MONO_ABS": 0.4, "EOS": 2.0,   "EOS_ABS": 0.1,
    "BASO": 0.5,  "BASO_ABS": 0.05, "MPV": 9.5,
}


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

_OCR_SYSTEM = """
You are a highly accurate medical data extraction assistant. Your primary function is to meticulously extract all numeric laboratory values and their corresponding exact test names from a given lab report image.

### CORE TASK: PRECISE NUMERIC EXTRACTION
1.  **Extract Numeric Values Only**: Identify and extract only the numerical values associated with each lab test. Disregard any units (e.g., g/dL, cells/µL, %) during the value extraction, as these will be handled by a downstream process. The extracted value MUST be a float or integer.
2.  **Exact Test Names**: Use the **exact test names** as they appear printed on the lab report image. Do not normalize, abbreviate, or alter these names in any way during extraction.
3.  **Handling Ambiguity**: If a numeric value for a test is unclear, unreadable, or cannot be confidently extracted, **DO NOT include that test-value pair** in the `extracted_values` output. It is better to omit uncertain data than to provide incorrect data.
4.  **Image Quality Assessment**: Provide concise notes on the overall quality of the image, highlighting any issues that might have impacted extraction accuracy (e.g., "blurry text", "poor lighting", "skewed document"). If the image quality is excellent, state "Image quality is excellent."

### OUTPUT REQUIREMENTS:
Return **ONLY** a raw JSON object. No markdown formatting (no ```json), no preamble, no conversational text, and no additional explanations.

### JSON STRUCTURE:
{
  "extracted_values": {
    "<Exact Test Name 1>": <float_value_1>,
    "<Exact Test Name 2>": <float_value_2>
  },
  "ocr_notes": "<Concise notes on image quality or extraction challenges>"
}
""".strip()


_MASTER_LLM_PROCESSOR = """
You are a Medical Data Architect specialized in Hematology. Your mission is to synchronize raw laboratory results into a standardized, multi-format JSON output for machine learning inference.

### INPUTS RECEIVED:
1.  **raw_ocr**: Unstructured numeric data extracted from a lab report image.
2.  **manual_data**: Numeric data provided directly by the user.
3.  **aliases**: A dictionary mapping medical terms to their canonical base keys.
4.  **normal_values**: Default biological reference values for the target columns.
5.  **target_columns**: The exact list of keys that **MUST** be present in the final output.

### CORE TASK: DATA HARMONIZATION & TRIPLE-MAPPING
1.  **Extraction & Conflict Resolution**:
    *   Combine 'raw_ocr' and 'manual_data'.
    *   If a specific test has different values in OCR and manual input, **YOU MUST CHOOSE THE MANUAL_DATA value**.
    *   Any such disagreement **MUST** be logged in the "conflicts" list.

2.  **Synonym Triple-Mapping (CRITICAL)**:
    The 'target_columns' contains the same medical data in three different naming conventions. **You MUST ensure they are identical**:
    *   **Group A (Short):** Hb, RBC, WBC, HCT, MCV, MCH, RDW, MONO, PLATELETS.
    *   **Group B (Abbreviations):** HGB, RBC, HCT, MCV, MCH, MCHC, RDW, PLT, MPV, WBC, NEUT_ABS, LYMP_ABS, MONO_ABS, EOS_ABS, BASO_ABS.
    *   **Group C (Full with Units):** Hemoglobin (g/dL), WBC (cells/µL), RBC (million/µL), Hematocrit (%), MCV (fL), MCH (pg), MCHC (g/dL), Platelet Count (cells/µL), RDW (%), Neutrophils (%), Lymphocytes (%), Monocytes (%).

3.  **Imputation & Zero-Null Policy**:
    *   For every key in 'target_columns', if no data is found, **use the value from 'normal_values'**.
    *   **Every single key MUST have a numeric float value. ABSOLUTELY NO NULLS, NO STRINGS, AND NO MISSING KEYS.**
    *   All imputed keys **MUST** be logged in the "imputed_keys" list.

### OUTPUT REQUIREMENTS:
Return **ONLY** a raw JSON object. No markdown formatting (no ```json), no preamble, no conversational text.

### JSON STRUCTURE:
{
  "final_data": {
    "HGB": 0.0, "Hb": 0.0, "WBC": 0.0, "RBC": 0.0, "HCT": 0.0,
    "MCV": 0.0, "MCH": 0.0, "MCHC": 0.0, "RDW": 0.0, "PLT": 0.0,
    "MPV": 0.0, "NEUT_ABS": 0.0, "LYMP_ABS": 0.0, "MONO": 0.0,
    "MONO_ABS": 0.0, "EOS_ABS": 0.0, "BASO_ABS": 0.0,
    "NEUT": 0.0, "LYMP": 0.0, "EOS": 0.0, "BASO": 0.0
  },
  "imputed_keys": [],
  "conflicts": []
}
""".strip()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_json_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text.strip(), re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def _run_ocr(image_value: str) -> tuple:
    if "," in image_value:
        header, data = image_value.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
    else:
        data      = image_value
        mime_type = "image/jpeg"

    raw_text = get_vision_response(_OCR_SYSTEM, data, mime_type)
    parsed   = _parse_json_response(raw_text)
    return parsed.get("extracted_values", {}), parsed.get("ocr_notes", "")


def _run_master_llm(ocr_data: dict, manual_data: dict) -> dict:
    payload = {
        "raw_ocr":        ocr_data,
        "manual_data":    manual_data,
        "aliases":        KEY_ALIASES,
        "normal_values":  NORMAL_VALUES,
        "target_columns": TARGET_COLUMNS,
    }
    prompt   = f"{_MASTER_LLM_PROCESSOR}\n\nINPUT DATA:\n{json.dumps(payload)}"
    raw_text = get_response(prompt)
    return _parse_json_response(raw_text)


# ──────────────────────────────────────────────
# Node Class
# ──────────────────────────────────────────────

class ExtractAndStandardizeNode:
    """
    Node 2: Runs OCR on lab image (if present) then calls the master LLM
    to produce a fully standardized dict of lab values. Returns dict slice.
    """

    def __call__(self, state: HematologyGraphState) -> dict:
        inp   = state["input_state"]
        agent = deepcopy(state["agent_state"])  # ✅ never mutate original

        warnings = list(agent.get("warnings") or [])
        errors   = list(agent.get("errors")   or [])

        # ── Skip if input was invalid ───────────────────────────────────
        if not agent.get("is_valid_input", False):
            agent["current_node"]   = "extract_and_standardize_node"
            agent["visited_nodes"]  = list(agent.get("visited_nodes") or []) + ["extract_and_standardize_node"]  # ✅ + not .append()
            agent["decision_trace"] = list(agent.get("decision_trace") or []) + ["extract_and_standardize_node:SKIPPED"]
            return {"agent_state": agent}  # ✅ return dict slice

        # ── 1. OCR (if lab report image exists) ────────────────────────
        ocr_data       = {}
        extracted_text = ""
        if inp.get("lab_report_image"):
            ocr_data, extracted_text = _run_ocr(inp["lab_report_image"])

        # ── 2. Master LLM ───────────────────────────────────────────────
        manual_data = inp.get("manual_lab_data") or {}
        llm_result  = _run_master_llm(ocr_data, manual_data)

        # ── 3. Extract results ──────────────────────────────────────────
        final_data   = llm_result.get("final_data",   {})
        imputed_keys = llm_result.get("imputed_keys", [])
        conflicts    = llm_result.get("conflicts",    [])

        if not final_data:
            errors.append("LLM_PROCESSING_FAILED: Could not generate valid standardized data.")
            agent["fallback_used"] = True

        if conflicts:
            warnings.extend([f"DATA_CONFLICT: {c}" for c in conflicts])
            agent["modality_conflict"] = True

        if imputed_keys:
            warnings.append(f"LLM_IMPUTED: {', '.join(imputed_keys)}")

        # ── 4. Data completeness ────────────────────────────────────────
        core_tests   = len(TARGET_COLUMNS)  # ✅ use actual count (21), not hardcoded 12
        real_tests   = max(0, core_tests - len(imputed_keys))
        completeness = round(real_tests / core_tests, 4)

        if completeness < 0.5:
            warnings.append(
                f"LOW_COMPLETENESS: completeness={completeness:.0%} - Most data was imputed."
            )

        # ── 5. Update state ─────────────────────────────────────────────
        agent.update({
            "current_node":      "extract_and_standardize_node",
            "visited_nodes":     list(agent.get("visited_nodes") or []) + ["extract_and_standardize_node"],  # ✅
            "decision_trace":    list(agent.get("decision_trace") or []) + ["extract_and_standardize_node"],
            "extracted_text":    extracted_text,
            "standardized_data": final_data,
            "data_completeness": completeness,
            "warnings":          warnings,
            "errors":            errors,
        })

        return {"agent_state": agent}  # ✅ return dict slice


# ──────────────────────────────────────────────
# Ready-to-use instance for LangGraph
# ──────────────────────────────────────────────
extract_and_standardize_node = ExtractAndStandardizeNode()