from typing import Literal
from langgraph.graph import StateGraph, END

# 1. استدعاء الـ State من الملف الأساسي (عشان ميبقاش فيه نسختين)
from state import HematologyGraphState

# 2. استيراد النودز (تأكد إن أسماء الملفات دي مطابقة للي عندك بالظبط)
from Nodes.Validate_Node import validate_inputs_node
from Nodes.Extract_and_Standardized_Node import extract_and_standardize_node
from Nodes.Binary_classify_Node import binary_classifier_node
from Nodes.disease_classifier_node import disease_classifier_node
from Nodes.severity_assessor_node import severity_assessor_node
from Nodes.Report_Generator import report_generator_node

# 3. تعريف دالة التوجيه الشرطي (Conditional Router)
def route_after_binary(state: HematologyGraphState) -> Literal["disease_classifier_node", "report_generator_node"]:
    agent = state["agent_state"]
    data = agent.get("standardized_data", {})
    
    # 1. استخراج القيم مع وضع قيم افتراضية آمنة (Normal Defaults)
    hgb  = data.get("HGB", 14.0)
    wbc  = data.get("WBC", 7.0)
    plt  = data.get("PLT", 250.0)
    mcv  = data.get("MCV", 90.0)
    rbc  = data.get("RBC", 4.5)
    neut = data.get("NEUT_ABS", 4.0)
    lymp = data.get("LYMP_ABS", 2.0)

    # 2. تعريف صمامات الأمان الطبية (Hard Medical Rules)
    # أي شرط من دول يتحقق = مريض فوراً بغض النظر عن رأي الموديل
    rules = [
        hgb < 9.0 or hgb > 18.0,           # أنيميا شديدة أو زيادة دم مفرطة
        wbc < 3.5 or wbc > 15.0,           # نقص مناعة حاد أو التهاب/لوكيميا
        plt < 100.0 or plt > 600.0,        # خطر نزيف أو تجلط عالي
        mcv < 75.0 or mcv > 105.0,         # خلل واضح في حجم الكريات
        rbc < 3.5 or rbc > 6.5,            # نقص أو زيادة حادة في الكريات الحمراء
        neut < 1.5,                        # نقص حاد في الخلايا المتعادلة (خطر عدوى)
        lymp > 5.0                         # زيادة مفرطة في الخلايا الليمفاوية
    ]

    is_medically_sick = any(rules)

    # 3. اتخاذ القرار
    # لو القواعد الطبية قالت مريض OR الموديل الباينري قال مريض
    if is_medically_sick or agent.get("is_sick") is True:
        print(f"🔴 Path: Patient Classified as SICK. (Medically Sick: {is_medically_sick}, AI Sick: {agent.get('is_sick')})")
        # نضمن إن الحالة تسمع في باقي النودز
        agent["is_sick"] = True 
        return "disease_classifier_node"
    
    # حالة السليم (لازم الاثنين يتفقوا إنه سليم)
    else:
        print("🟢 Path: Patient is Healthy. Routing to Final Report...")
        return "report_generator_node"
# 4. بناء الجراف وتجميع القطع
def build_hematology_graph():
    # إنشاء جراف يعتمد على الـ State اللي عرفناه
    workflow = StateGraph(HematologyGraphState)

    # --- أ. إضافة النودز (العمال) ---
    workflow.add_node("validate_inputs_node", validate_inputs_node)
    workflow.add_node("extract_and_standardize_node", extract_and_standardize_node)
    workflow.add_node("binary_classifier_node", binary_classifier_node)
    workflow.add_node("disease_classifier_node", disease_classifier_node)
    workflow.add_node("severity_assessor_node", severity_assessor_node)
    workflow.add_node("report_generator_node", report_generator_node)

    # --- ب. ربط المسارات (Edges) ---
    
    # البداية الإجبارية: فحص المدخلات -> استخراج الداتا -> التصنيف المبدئي
    workflow.set_entry_point("validate_inputs_node")
    workflow.add_edge("validate_inputs_node", "extract_and_standardize_node")
    workflow.add_edge("extract_and_standardize_node", "binary_classifier_node")

    # مفترق الطرق: بعد نود البينري، السيستم هيقرأ قيمة is_sick ويقرر
    workflow.add_conditional_edges(
        "binary_classifier_node",
        route_after_binary,
        {
            "disease_classifier_node": "disease_classifier_node",
            "report_generator_node": "report_generator_node"
        }
    )

    # مسار المريض (لو مريض): تحديد المرض -> تقييم الخطورة -> التقرير النهائي
    workflow.add_edge("disease_classifier_node", "severity_assessor_node")
    workflow.add_edge("severity_assessor_node", "report_generator_node")

    # النهاية الإجبارية
    workflow.add_edge("report_generator_node", END)

    # --- ج. تجميع الجراف (Compile) ---
    return workflow.compile()

# نعمل Instance جاهز للاستخدام في الـ FastAPI بعدين
hematology_agent = build_hematology_graph()