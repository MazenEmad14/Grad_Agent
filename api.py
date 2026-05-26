"""
╔══════════════════════════════════════════════════════════════════╗
║          Hematology AI Agent — FastAPI Server                   ║
║  Exposes the LangGraph diagnostic pipeline via REST endpoints.  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import base64
import uuid
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Internal imports ──────────────────────────────────────────────
from graph_builder import hematology_agent
from Nodes.Binary_classify_Node import BinaryClassificationEngine

# ── Bootstrap ─────────────────────────────────────────────────────
load_dotenv()

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="🩸 Hematology AI Diagnostic API",
    description=(
        "A Clinical Decision Support System (CDSS) that analyses blood smear images "
        "and/or lab reports to detect and classify hematological diseases using a "
        "multi-modal LangGraph AI pipeline."
    ),
    version="1.0.0",
    contact={
        "name": "Graduation Project Team",
    },
    license_info={"name": "MIT"},
)

# ── CORS (allow any origin — tighten in production) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton ML Engine (loaded once at startup) ──────────────────
engine: Optional[BinaryClassificationEngine] = None


@app.on_event("startup")
async def startup_event():
    global engine
    print("🚀 Loading Binary Classification Engine…")
    engine = BinaryClassificationEngine(models_dir="models/")
    print("✅ Engine ready.")


# ══════════════════════════════════════════════════════════════════
#  Pydantic schemas
# ══════════════════════════════════════════════════════════════════

class ManualLabData(BaseModel):
    """Optional structured lab values (CBC panel)."""
    HGB:      Optional[float] = Field(None, description="Hemoglobin g/dL")
    WBC:      Optional[float] = Field(None, description="White blood cells ×10³/µL")
    PLT:      Optional[float] = Field(None, description="Platelets ×10³/µL")
    RBC:      Optional[float] = Field(None, description="Red blood cells ×10⁶/µL")
    MCV:      Optional[float] = Field(None, description="Mean corpuscular volume fL")
    MCH:      Optional[float] = Field(None, description="Mean corpuscular hemoglobin pg")
    MCHC:     Optional[float] = Field(None, description="MCHC g/dL")
    HCT:      Optional[float] = Field(None, description="Hematocrit %")
    NEUT_ABS: Optional[float] = Field(None, description="Neutrophils absolute ×10³/µL")
    LYMP_ABS: Optional[float] = Field(None, description="Lymphocytes absolute ×10³/µL")
    MONO_ABS: Optional[float] = Field(None, description="Monocytes absolute ×10³/µL")
    EOS_ABS:  Optional[float] = Field(None, description="Eosinophils absolute ×10³/µL")
    BASO_ABS: Optional[float] = Field(None, description="Basophils absolute ×10³/µL")


class DiagnosisResponse(BaseModel):
    """Unified response schema for all diagnostic endpoints."""
    request_id:    str
    elapsed_ms:    float

    # ── Core result ─────────────────────────────────────────────
    is_sick:           Optional[bool]
    disease_type:      Optional[str]
    severity_level:    Optional[str]
    risk_level:        Optional[str]
    disease_confidence: float
    urgent_action_required: bool

    # ── Quality signals ──────────────────────────────────────────
    low_confidence_flag: bool
    modality_conflict:   bool
    data_completeness:   float
    missing_modalities:  list

    # ── Outputs ──────────────────────────────────────────────────
    standardized_data:     Dict[str, Any]
    recommendations:       Dict[str, Any]
    final_report:          Optional[str]
    warnings:              list
    decision_trace:        list


# ══════════════════════════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════════════════════════

def _file_to_base64(file_bytes: bytes, content_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def _build_initial_state(
    blood_smear_b64: Optional[str],
    lab_report_b64: Optional[str],
    manual_lab: Optional[Dict[str, float]],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    return {
        "input_state": {
            "session_id":        session_id or str(uuid.uuid4()),
            "user_id":           user_id,
            "blood_smear_image": blood_smear_b64,
            "lab_report_image":  lab_report_b64,
            "manual_lab_data":   manual_lab,
            "input_type":        "unknown",
        },
        "agent_state": {
            "visited_nodes":  [],
            "decision_trace": [],
            "errors":         [],
            "warnings":       [],
            "retry_count":    0,
        },
    }


def _run_pipeline(initial_state: dict) -> dict:
    """Invoke the LangGraph agent and return the agent_state dict."""
    if engine is None:
        raise RuntimeError("ML engine not initialised yet.")
    config = {"configurable": {"binary_engine": engine}}
    result = hematology_agent.invoke(initial_state, config=config)
    return result["agent_state"]


def _build_response(agent: dict, request_id: str, elapsed_ms: float) -> DiagnosisResponse:
    return DiagnosisResponse(
        request_id=request_id,
        elapsed_ms=round(elapsed_ms, 2),
        is_sick=agent.get("is_sick"),
        disease_type=agent.get("disease_type"),
        severity_level=agent.get("severity_level"),
        risk_level=agent.get("risk_level"),
        disease_confidence=agent.get("disease_confidence", 0.0),
        urgent_action_required=bool(agent.get("requires_doctor") or agent.get("critical_flags")),
        low_confidence_flag=bool(agent.get("low_confidence_flag", False)),
        modality_conflict=bool(agent.get("modality_conflict", False)),
        data_completeness=agent.get("data_completeness", 0.0),
        missing_modalities=agent.get("missing_modalities", []),
        standardized_data=agent.get("standardized_data", {}),
        recommendations=agent.get("recommendations", {}),
        final_report=agent.get("final_report"),
        warnings=agent.get("warnings", []),
        decision_trace=agent.get("decision_trace", []),
    )


# ══════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════

# ── Health check ──────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Hematology AI Diagnostic API",
        "version": "1.0.0",
        "status":  "running",
        "engine_ready": engine is not None,
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "engine_ready": engine is not None}


# ── 1. Diagnose via Blood Smear Only (Vision) ─────────────────────
@app.post(
    "/diagnose/blood-smear",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnosis Paths"],
    summary="Path 1: Diagnose via Blood Smear Image Only",
)
async def diagnose_blood_smear(
    blood_smear_image: UploadFile = File(..., description="صورة شريحة الدم المجهرية"),
    session_id: Optional[str] = Form(None)
):
    """
    استخدم هذه البوابة لرفع **صورة خلية الدم المجهرية** فقط.
    سيتم توجيهها تلقائياً لموديل الرؤية الحاسوبية (Vision-Only).
    """
    data = await blood_smear_image.read()
    smear_b64 = _file_to_base64(data, blood_smear_image.content_type or "image/jpeg")
    
    initial_state = _build_initial_state(
        blood_smear_b64=smear_b64, 
        lab_report_b64=None, 
        manual_lab=None, 
        session_id=session_id
    )
    
    t0 = time.perf_counter()
    try:
        agent = _run_pipeline(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
        
    return _build_response(agent, str(uuid.uuid4()), (time.perf_counter() - t0) * 1000)


# ── 2. Diagnose via Lab Report Only (OCR & Tabular) ───────────────
@app.post(
    "/diagnose/lab-report",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnosis Paths"],
    summary="Path 2: Diagnose via Lab Report Image Only",
)
async def diagnose_lab_report(
    lab_report_image: UploadFile = File(..., description="صورة تقرير المعمل (الورقة)"),
    session_id: Optional[str] = Form(None)
):
    """
    استخدم هذه البوابة لرفع **صورة ورقة تحليل الـ CBC** فقط.
    سيتم قراءة الأرقام (OCR) وتوجيهها لموديل البيانات المجدولة (Tabular-Only).
    """
    data = await lab_report_image.read()
    report_b64 = _file_to_base64(data, lab_report_image.content_type or "image/jpeg")
    
    initial_state = _build_initial_state(
        blood_smear_b64=None, 
        lab_report_b64=report_b64, 
        manual_lab=None, 
        session_id=session_id
    )
    
    t0 = time.perf_counter()
    try:
        agent = _run_pipeline(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
        
    return _build_response(agent, str(uuid.uuid4()), (time.perf_counter() - t0) * 1000)


# ── 3. Diagnose via Fusion  ──────────────────────────
@app.post(
    "/diagnose/fusion",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnosis Paths"],
    summary="Path 3: Full Fusion (Smear Image + Lab Report Image OR Manual Values)",
)
async def diagnose_fusion(
    blood_smear_image: UploadFile = File(..., description="صورة شريحة الدم المجهرية (إلزامية)"),
    lab_report_image: Optional[UploadFile] = File(None, description="صورة ورقة تحليل المعمل (اختيارية لو هتدخل القيم يدوياً)"),
    # إتاحة إدخال القيم يدوياً كـ Form Fields في نفس الوقت
    HGB:      Optional[float] = Form(None),
    WBC:      Optional[float] = Form(None),
    PLT:      Optional[float] = Form(None),
    RBC:      Optional[float] = Form(None),
    MCV:      Optional[float] = Form(None),
    MCH:      Optional[float] = Form(None),
    MCHC:     Optional[float] = Form(None),
    HCT:      Optional[float] = Form(None),
    NEUT_ABS: Optional[float] = Form(None),
    LYMP_ABS: Optional[float] = Form(None),
    MONO_ABS: Optional[float] = Form(None),
    EOS_ABS:  Optional[float] = Form(None),
    BASO_ABS: Optional[float] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """
    بوابة الدمج الكامل:
    يجب رفع صورة خلية الدم، ومعها إما (صورة ورقة التحليل ليقرأها الذكاء الاصطناعي) أو (كتابة أرقام التحاليل يدوياً).
    """
    # 1. تجميع البيانات اليدوية لو موجودة
    manual_raw = {
        k: v for k, v in {
            "HGB": HGB, "WBC": WBC, "PLT": PLT, "RBC": RBC,
            "MCV": MCV, "MCH": MCH, "MCHC": MCHC, "HCT": HCT,
            "NEUT_ABS": NEUT_ABS, "LYMP_ABS": LYMP_ABS,
            "MONO_ABS": MONO_ABS, "EOS_ABS": EOS_ABS, "BASO_ABS": BASO_ABS,
        }.items() if v is not None
    }
    manual = manual_raw if manual_raw else None

    # صمام أمان: التأكد إن المستخدم بعت يا إما صورة تحليل يا إما قيم يدوية مع صورة الخلية
    if lab_report_image is None and not manual:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="لإتمام عملية الدمج (Fusion)، يجب رفع صورة ورقة التحليل أو إدخال قيم CBC يدوياً مع صورة الخلية.",
        )

    t0 = time.perf_counter()

    # 2. تحويل صورة الخلية لـ Base64
    smear_data = await blood_smear_image.read()
    smear_b64 = _file_to_base64(smear_data, blood_smear_image.content_type or "image/jpeg")

    # 3. تحويل صورة التحليل لـ Base64 لو موجودة
    report_b64 = None
    if lab_report_image:
        report_data = await lab_report_image.read()
        report_b64 = _file_to_base64(report_data, lab_report_image.content_type or "image/jpeg")

    # 4. بناء الـ Initial State وضخ البيانات للجراف
    initial_state = _build_initial_state(
        blood_smear_b64=smear_b64, 
        lab_report_b64=report_b64, 
        manual_lab=manual, 
        session_id=session_id
    )
    
    try:
        agent = _run_pipeline(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
        
    return _build_response(agent, str(uuid.uuid4()), (time.perf_counter() - t0) * 1000)

# ── 4. Diagnose with manual lab data only (Fallback) ───────────────
@app.post(
    "/diagnose/manual",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnosis Paths"],
    summary="Path 4: Diagnose using manual CBC values",
)
async def diagnose_manual(lab_data: ManualLabData):
    """
    Submit **only structured CBC lab values** manually (no images required).
    """
    values = lab_data.model_dump(exclude_none=True)
    if not values:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one lab value must be provided.",
        )

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    initial_state = _build_initial_state(
        blood_smear_b64=None,
        lab_report_b64=None,
        manual_lab=values,
    )

    try:
        agent = _run_pipeline(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    return _build_response(agent, request_id, (time.perf_counter() - t0) * 1000)


# ══════════════════════════════════════════════════════════════════
#  Entry point  (python api.py)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # hot-reload during development
        log_level="info",
    )