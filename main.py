import os
import base64
from dotenv import load_dotenv
import pprint

# استيراد الشغل بتاعنا
from graph_builder import hematology_agent  # استدعاء الـ Compiled Graph مباشرة
from Nodes.Binary_classify_Node import BinaryClassificationEngine

# تحميل مفاتيح البيئة (API Key)
load_dotenv()

def run_diagnostic_pipeline(input_data: dict, engine_instance):
    """
    الدالة الرئيسية لتشغيل رحلة الفحص كاملة.
    """
    # تجهيز الحالة الابتدائية
    initial_state = {
        "input_state": {
            "blood_smear_image": input_data.get("blood_smear_image"),
            "lab_report_image": input_data.get("lab_report_image"),
            "manual_lab_data": input_data.get("manual_lab_data"),
            "input_type": "unknown" 
        },
        "agent_state": {
            "visited_nodes": [],
            "decision_trace": [],
            "errors": [],
            "warnings": [],
            "retry_count": 0
        }
    }
    
    # تشغيل الجراف
    # بنبعت الـ engine جوه الـ config عشان النودز تقدر تستخدمه
    config = {"configurable": {"binary_engine": engine_instance}}
    
    final_output = hematology_agent.invoke(initial_state, config=config)
    return final_output["agent_state"]

# دالة لتحويل الصورة لـ Base64
def image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

# ==========================================
# 🧪 تجربة دمج (صورة الفيلم + صورة التحليل الورقي)
# ==========================================
if __name__ == "__main__":
    # 1. تهيئة المحرك
    engine = BinaryClassificationEngine(models_dir="models/")
    
    # 2. تحويل الصور لـ Base64
    # تأكد إن الملفين test_image.jpg و images.jpeg موجودين في نفس الفولدر
    smear_image_b64 = image_to_base64("test_image.jpg") 
    lab_report_b64  = image_to_base64("images.jpeg")

    if not smear_image_b64 or not lab_report_b64:
        print("❌ خطأ: تأكد من وجود الملفات test_image.jpg و images.jpeg")
    else:
        # 3. تجهيز الداتا المختلطة
        multimodal_data = {
            "blood_smear_image": smear_image_b64, # صورة الفيلم (المجهر)
            "lab_report_image": lab_report_b64,   # صورة التحليل (الورقة)
            "manual_lab_data": None                # سيبها None عشان يعتمد على الـ OCR
        }

        print("\n" + "🧬" * 10)
        print("RUNNING MULTIMODAL TEST: (Smear Image + Lab Report Image)")
        print("🧬" * 10)
        
        try:
            # تشغيل الجراف بالكامل
            result = run_diagnostic_pipeline(multimodal_data, engine)
            
            print("\n" + "="*60)
            print("🩺 FINAL MULTIMODAL DIAGNOSIS")
            print("="*60)
            print(f"📍 Disease:    {result.get('disease_type')}")
            print(f"📍 Severity:   {result.get('severity_level')}")
            print(f"📍 Confidence: {result.get('disease_confidence', 0) * 100:.2f}%")
            print(f"🚨 Urgent:     {result.get('urgent_action_required')}")
            
            print("\n📊 DATA EXTRACTED FROM images.jpeg:")
            pprint.pprint(result.get("standardized_data", {}))

            print("\n⚠️ MODALITY CONFLICT:")
            print(f"❓ Is there a conflict? {result.get('modality_conflict', False)}")
            if result.get("warnings"):
                print(f"⚠️ Warnings: {result.get('warnings')}")

            print("\n📋 REPORT HIGHLIGHTS:")
            print(result.get("final_report", "")[:400] + "...")

            print("\n🛤️ PATH TAKEN:")
            print(" -> ".join(result.get("decision_trace", [])))

        except Exception as e:
            print(f"❌ Multimodal Pipeline Failed: {str(e)}")