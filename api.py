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


class Base64DiagnoseRequest(BaseModel):
    """Send pre-encoded images as base64 strings."""
    blood_smear_image: Optional[str] = Field(
        None,
        description="Blood smear microscopy image encoded as base64 (data URI or raw base64).",
    )
    lab_report_image: Optional[str] = Field(
        None,
        description="Lab report scan encoded as base64.",
    )
    manual_lab_data: Optional[ManualLabData] = Field(
        None,
        description="Manually entered CBC values (overrides OCR when provided).",
    )
    session_id: Optional[str] = Field(None, description="Optional session identifier.")
    user_id:    Optional[str] = Field(None, description="Optional user identifier.")


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

def _ensure_data_uri(b64: Optional[str]) -> Optional[str]:
    """Guarantee the string is a proper data URI."""
    if b64 is None:
        return None
    if b64.startswith("data:"):
        return b64
    return f"data:image/jpeg;base64,{b64}"


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
    """API root — returns status info."""
    return {
        "service": "Hematology AI Diagnostic API",
        "version": "1.0.0",
        "status":  "running",
        "engine_ready": engine is not None,
    }


@app.get("/health", tags=["Health"])
async def health():
    """Liveness probe."""
    return {"status": "ok", "engine_ready": engine is not None}


# ── Diagnose via base64 ───────────────────────────────────────────

@app.post(
    "/diagnose/base64",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnosis"],
    summary="Diagnose using base64-encoded images",
)
async def diagnose_base64(payload: Base64DiagnoseRequest):
    """
    Submit blood-smear and/or lab-report images as **base64** strings,
    along with optional manually entered CBC values.

    At least one of `blood_smear_image`, `lab_report_image`, or
    `manual_lab_data` must be provided.
    """
    if not any([
        payload.blood_smear_image,
        payload.lab_report_image,
        payload.manual_lab_data,
    ]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one of: blood_smear_image, lab_report_image, manual_lab_data.",
        )

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    manual = payload.manual_lab_data.model_dump(exclude_none=True) if payload.manual_lab_data else None

    initial_state = _build_initial_state(
        blood_smear_b64=_ensure_data_uri(payload.blood_smear_image),
        lab_report_b64=_ensure_data_uri(payload.lab_report_image),
        manual_lab=manual,
        session_id=payload.session_id,
        user_id=payload.user_id,
    )

    try:
        agent = _run_pipeline(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    elapsed = (time.perf_counter() - t0) * 1000
    return _build_response(agent, request_id, elapsed)


# ── Diagnose via file upload ──────────────────────────────────────

@app.post(
    "/diagnose/upload",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnosis"],
    summary="Diagnose using multipart file upload",
)
async def diagnose_upload(
    blood_smear_image: Optional[UploadFile] = File(None,  description="Blood smear microscopy image file."),
    lab_report_image:  Optional[UploadFile] = File(None,  description="Lab report scan image file."),
    # Manual CBC values as form fields
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
    session_id: Optional[str] = Form(None),
    user_id:    Optional[str] = Form(None),
):
    """
    Submit images as **multipart/form-data** file uploads.
    CBC lab values can also be passed as optional form fields.

    At least one image or one lab value is required.
    """
    # Collect manual lab values
    manual_raw = {
        k: v for k, v in {
            "HGB": HGB, "WBC": WBC, "PLT": PLT, "RBC": RBC,
            "MCV": MCV, "MCH": MCH, "MCHC": MCHC, "HCT": HCT,
            "NEUT_ABS": NEUT_ABS, "LYMP_ABS": LYMP_ABS,
            "MONO_ABS": MONO_ABS, "EOS_ABS": EOS_ABS, "BASO_ABS": BASO_ABS,
        }.items() if v is not None
    }
    manual = manual_raw if manual_raw else None

    if blood_smear_image is None and lab_report_image is None and not manual:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Upload at least one image or supply lab values.",
        )

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    # Convert uploaded files to base64
    smear_b64 = None
    if blood_smear_image:
        data = await blood_smear_image.read()
        smear_b64 = _file_to_base64(data, blood_smear_image.content_type or "image/jpeg")

    report_b64 = None
    if lab_report_image:
        data = await lab_report_image.read()
        report_b64 = _file_to_base64(data, lab_report_image.content_type or "image/jpeg")

    initial_state = _build_initial_state(
        blood_smear_b64=smear_b64,
        lab_report_b64=report_b64,
        manual_lab=manual,
        session_id=session_id,
        user_id=user_id,
    )

    try:
        agent = _run_pipeline(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    elapsed = (time.perf_counter() - t0) * 1000
    return _build_response(agent, request_id, elapsed)


# ── Diagnose with manual lab data only ───────────────────────────

@app.post(
    "/diagnose/manual",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnosis"],
    summary="Diagnose using manually entered CBC lab values only",
)
async def diagnose_manual(lab_data: ManualLabData):
    """
    Submit **only structured CBC lab values** (no images required).

    Useful when there is no image available but lab results are known.
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

    elapsed = (time.perf_counter() - t0) * 1000
    return _build_response(agent, request_id, elapsed)


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
