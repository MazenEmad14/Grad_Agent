from typing import TypedDict, Optional, Dict, List

# ==========================================
# 1. Input State (Immutable data from App)
# ==========================================
class InputState(TypedDict):
    """
    Represents the initial payload provided by the user.
    """
    # ===== System Context =====
    session_id: Optional[str]
    user_id: Optional[str]
    
    # ===== Input Definition =====
    input_type: str  # e.g., 'image_only', 'data_only', or 'both'
    
    # ===== Raw Inputs =====
    blood_smear_image: Optional[str]             # Base64 or URL
    lab_report_image: Optional[str]              # Base64 or URL
    manual_lab_data: Optional[Dict[str, float]]  # e.g., {'Hb': 10.5}


# ==========================================
# 2. Agent State (Mutable reasoning state)
# ==========================================
class AgentState(TypedDict):
    """
    Internal AI memory. Lists are initialized in the first node to avoid Optional checks.
    """
    # ===== Control Flow (Dynamic) =====
    current_node: Optional[str]
    visited_nodes: List[str]            # Initialized to []
    decision_trace: List[str]           # Initialized to []

    # ===== Configuration (Constants) =====
    model_versions: Dict[str, str]      # e.g., {'cnn': 'v1'}
    confidence_threshold: float         # default = 0.6
    topic: Optional[str]
    
    # ===== Patient History =====
    patient_history: List[Dict]         # Previous results
    trend_timestamps: List[str]
    trend_analysis: Optional[str]

    # ===== Validation & Robustness =====
    is_valid_input: bool
    validation_errors: List[str]        # Initialized to []
    errors: List[str]                   # Initialized to []
    warnings: List[str]                 # Initialized to []
    retry_count: int
    fallback_used: bool

    # ===== Medical Knowledge Context =====
    reference_ranges: Dict[str, Dict[str, float]]
    rule_flags: List[str]               # Initialized to []

    # ===== Data Quality & Processing =====
    data_completeness: float            # if < 0.5, triggers low_confidence_flag
    extracted_text: Optional[str]
    cleaned_data: Dict[str, float]
    standardized_data: Dict[str, float]
    missing_modalities: List[str]       # Initialized to [] (e.g., ['lab_data'])

    # ===== ML Models (Output Hub) =====
    is_sick: Optional[bool]
    disease_type: Optional[str]
    disease_confidence: float
    disease_candidates: List[Dict[str, float]]
    
    severity_level: Optional[str]
    risk_level: Optional[str]

    # ===== Internal AI Insights (Explainability Only) =====
    # Renamed to clarify these are NOT for routing decisions
    fusion_insight: Optional[str]       
    fusion_insight_confidence: Optional[float]
    modality_conflict: bool             # Conflict between image and data signals
    low_confidence_flag: bool           # Triggered by threshold or low completeness
    explanations: Dict[str, str]

    # ===== Decision & Emergency Layer =====
    requires_doctor: bool
    critical_flags: List[str]           # Initialized to []
    needs_additional_testing: bool

    # ===== Final Output & UX =====
    final_diagnosis: Optional[str]
    confidence_level: Optional[str]     # High, Medium, Low
    recommendations: Dict[str, List[str]]
    final_recommendations: Optional[str]
    final_report: Optional[str]


# ==========================================
# 3. Main Graph State (The complete payload)
# ==========================================
class HematologyGraphState(TypedDict):
    """
    The object passed between LangGraph nodes.
    """
    input_state: InputState
    agent_state: AgentState