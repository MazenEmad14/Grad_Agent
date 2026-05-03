"""
Node 6: ReportGeneratorNode
──────────────────────────────────────
المسؤولية:
  - أخذ كل حالة المريض (Data, Diagnosis, Severity, Warnings).
  - إرسالها للـ LLM لتوليد تقرير طبي شامل ومخصص للحالة.
  - إرجاع التقرير بصيغة Markdown جاهزة للعرض في الـ UI.
"""

import json
import re
from copy import deepcopy

from state import HematologyGraphState
from llm import get_response


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

_REPORT_GENERATOR_PROMPT = """
You are an Expert Hematologist and AI Medical Assistant. Your task is to write a final, comprehensive, and patient-friendly medical analysis based on the provided AI pipeline state.

### PATIENT AI STATE:
{patient_state_json}

### INSTRUCTIONS FOR THE REPORT & RECOMMENDATIONS:

1.  **Language & Medical Terms**:
    * Write the entire content in **clear, professional Arabic**. Maintain a respectful, clinical, and professional tone.
    * **CRITICAL**: DO NOT use English abbreviations for lab tests (e.g., HGB, WBC, MCV) in the text. Translate them to their full Arabic medical names (e.g., "الهيموجلوبين", "خلايا الدم البيضاء", "متوسط حجم الكرية الحمراء").

2.  **Report Content based on 'is_sick' status**:
    * **If 'is_sick' is False (Healthy)**:
        * **Reassurance**: Reassure the patient that their blood counts are within normal limits.
        * **Highlights**: Briefly highlight 2-3 main healthy values to reinforce the positive outcome.
        * **Follow-up**: Advise routine check-ups.
    * **If 'is_sick' is True (Sick)**:
        * **Diagnosis**: Clearly state the AI's suspected diagnosis (`disease_type`).
        * **Reasoning**: Explain the reasoning (`disease_reasoning`) in simple, understandable Arabic.
        * **Severity**: Address the `severity_level`. If 'High' or 'Critical', convey urgency without panic.

3.  **Recommendations (Actionable & Specific)**:
    * Provide 3-5 highly specific, actionable lifestyle, dietary, or medical recommendations.
    * If `is_sick` is True, tailor them directly to the `disease_type`.
    * If `is_sick` is False, provide general tips to maintain healthy blood.

4.  **Doctor's Notes (Data Warnings)**:
    * If `modality_conflict` is True OR if `system_warnings` contains any data conflicts or imputations,
      include a dedicated section titled "**ملاحظة للطبيب المعالج:**".
    * Briefly mention the detected conflicts or imputations concisely and technically.

5.  **Medical Disclaimer**:
    * You **MUST** end the report with a bold disclaimer in Arabic stating that this is an AI-generated
      report and MUST be verified by a certified doctor.

### CRITICAL JSON FORMATTING RULES:
1. Return **ONLY** a raw, strictly valid JSON object.
2. **DO NOT** wrap the JSON in Markdown blocks (no ```json). Start with { and end with }.
3. **DO NOT** use actual line breaks inside string values. Use \\n for newlines.
4. Escape all double quotes inside strings using \\" to prevent breaking the JSON.

### OUTPUT FORMAT:
{
  "patient_report_file": {
    "report_markdown": "## التقرير الطبي\\n\\n...(full report using \\n for newlines)...",
    "urgent_action_required": false,
    "doctor_summary": "<2-sentence technical summary for a doctor in Arabic>"
  },
  "patient_recommendations_file": {
    "title": "خطة الرعاية والتوصيات",
    "recommendations_list": [
      "<Recommendation 1 in Arabic>",
      "<Recommendation 2 in Arabic>",
      "<Recommendation 3 in Arabic>"
    ]
  }
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
        except Exception as e:
            print(f"JSON Parse Error in Report Node: {e}")
    return {}


# ──────────────────────────────────────────────
# Node Class
# ──────────────────────────────────────────────

class ReportGeneratorNode:
    """
    Node 6: Generates the final patient-facing medical report using an LLM.
    Reads all diagnosis, severity, and warning data set by upstream nodes.
    """

    def __call__(self, state: HematologyGraphState) -> dict:
        agent  = deepcopy(state["agent_state"])  # ✅ never mutate original
        errors = list(agent.get("errors") or [])  # ✅ safe — no crash if None

        # ── 1. Build patient context for the LLM ───────────────────────
        patient_context = {
            "lab_results":       agent.get("standardized_data", {}),
            "is_sick":           agent.get("is_sick", False),
            "disease_type":      agent.get("disease_type",      "None"),
            "disease_reasoning": agent.get("disease_reasoning", "None"),
            "severity_level":    agent.get("severity_level",    "None"),
            "system_warnings":   agent.get("warnings",          []),
            "modality_conflict": agent.get("modality_conflict", False),
            "ai_confidence":     agent.get("disease_confidence", 0.0),
        }

        # ── 2. Build prompt ─────────────────────────────────────────────
        prompt = _REPORT_GENERATOR_PROMPT.replace(
            "{patient_state_json}",
            json.dumps(patient_context, indent=2, ensure_ascii=False),
        )

        # ── 3. Call LLM (Protected with Try-Except) ─────────────────────
        print("✍️ Generating Final Medical Report...")
        try:
            raw_response = get_response(prompt)
            parsed_data  = _parse_json_response(raw_response)
        except Exception as e:
            print(f"❌ LLM Error: {str(e)}")
            errors.append(f"Report Generation Failed: {str(e)}")
            parsed_data = {}

        # ── 4. Fallback if LLM returned invalid JSON ────────────────────
        if not parsed_data:
            if not any("Report Generation Failed" in err for err in errors):
                errors.append("Report Generation Failed: Invalid LLM Response.")  # ✅ no crash
            parsed_data = {}

        # ── 5. Extract output — handle both nested and flat responses ───
        report_file = parsed_data.get("patient_report_file", {})
        recs_file   = parsed_data.get("patient_recommendations_file", {})

        final_markdown = (
            report_file.get("report_markdown")
            or parsed_data.get("report_markdown",
               "## ⚠️ عذرًا\nلم يتم توليد التقرير بشكل صحيح بسبب مشكلة في الخادم. يرجى مراجعة الطبيب.")
        )
        urgent_action = (
            report_file.get("urgent_action_required")
            or parsed_data.get("urgent_action_required", False)
        )
        doc_summary = (
            report_file.get("doctor_summary")
            or parsed_data.get("doctor_summary", "لا يوجد ملخص متاح حالياً.")
        )
        recs_title = (
            recs_file.get("title")
            or parsed_data.get("title", "خطة الرعاية والتوصيات")
        )
        recs_list = (
            recs_file.get("recommendations_list")
            or parsed_data.get("recommendations_list", [])
        )

        # ── 6. Update state ─────────────────────────────────────────────
        # ── 6. Update state ─────────────────────────────────────────────
        agent.update({
            "current_node":           "report_generator_node",
            "visited_nodes":          list(agent.get("visited_nodes") or []) + ["report_generator_node"],
            "decision_trace":         list(agent.get("decision_trace") or []) + ["report_generator_node"],
            "errors":                 errors,
            "final_report":           final_markdown,
            "urgent_action_required": urgent_action,
            "doctor_summary":         doc_summary,
            "recommendations_title":  recs_title,
            "recommendations_list":   recs_list,
        })

        # ── 7. Save to JSON File (For Database) ─────────────────────────
        import os
        import datetime
        
        # نعمل فولدر اسمه 'patient_records' لو مش موجود
        os.makedirs("patient_records", exist_ok=True)
        
        # نجهز الداتا اللي الداتابيز محتاجاها بس (من غير دوشة الـ LangGraph)
        db_record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "is_sick": agent.get("is_sick"),
            "disease_type": agent.get("disease_type"),
            "severity_level": agent.get("severity_level"),
            "ai_confidence": agent.get("disease_confidence"),
            "urgent_action_required": urgent_action,
            "lab_results": agent.get("standardized_data", {}),
            "doctor_summary": doc_summary,
            "patient_report": final_markdown,
            "recommendations": recs_list,
            "warnings": agent.get("warnings", [])
        }
        
        # نسمي الملف باسم الوقت عشان ميتكررش
        file_name = f"patient_records/record_{db_record['timestamp']}.json"
        
        try:
            # نحفظ الملف بصيغة JSON بتدعم العربي
            with open(file_name, "w", encoding="utf-8") as json_file:
                json.dump(db_record, json_file, ensure_ascii=False, indent=4)
            print(f"💾 Patient record saved successfully to: {file_name}")
        except Exception as e:
            print(f"❌ Failed to save JSON file: {e}")
            agent["errors"].append(f"JSON Save Error: {e}")

        return {"agent_state": agent}  # ✅ return dict slice


# ──────────────────────────────────────────────
# Ready-to-use instance for LangGraph
# ──────────────────────────────────────────────
report_generator_node = ReportGeneratorNode()