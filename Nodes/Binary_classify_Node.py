import os
import io
import base64
import numpy as np
import joblib
from copy import deepcopy
from typing import Tuple
from PIL import Image

# 🚨 مفيش TensorFlow هنا تاني، أهلًا PyTorch!
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from langchain_core.runnables import RunnableConfig

# =================================================================
# 1. PyTorch Model Architectures (بنفس هيكل النوت بوك بالمللي)
# =================================================================

class MultimodalFusion(nn.Module):
    def __init__(self, tab_dim):
        super(MultimodalFusion, self).__init__()
        # pretrained=False لأننا كدة كدة هنحمل أوزاننا ومش محتاجين ننزل من النت
        self.backbone = torchvision.models.efficientnet_b0(pretrained=False)
        self.backbone.classifier = nn.Identity()
        
        self.vision_head = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.6)
        )
        
        self.tab_head = nn.Sequential(
            nn.Linear(tab_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4)
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, img, tab):
        B, T, C, H, W = img.shape
        img = img.view(B * T, C, H, W)
        img_features = self.backbone(img)
        img_features = img_features.view(B, T, -1).mean(dim=1)
        x_v = self.vision_head(img_features)
        x_t = self.tab_head(tab)
        merged = torch.cat((x_v, x_t), dim=1)
        return self.fusion_head(merged)


class VisionOnlyResNet(nn.Module):
    def __init__(self):
        super(VisionOnlyResNet, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, img):
        B, T, C, H, W = img.shape
        img = img.view(B * T, C, H, W)
        features = self.backbone(img)
        features = features.view(B, T, -1).mean(dim=1)
        return self.head(features)


# =================================================================
# Engine: PyTorch Engine Loading Models
# =================================================================
class BinaryClassificationEngine:
    def __init__(self, models_dir: str = "models/"):
        print("🚀 Booting up PyTorch Engine and loading weights... This happens only once!")
        
        # اختيار الـ CPU أو GPU للسيرفر
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ Engine running on: {self.device}")
        
        # 🚨 الـ 15 عمود الصح عالنضافة بدون Status!
        self.tabular_feature_order = [
            "HGB", "RBC", "HCT", "MCV", "MCH", "MCHC", "RDW", 
            "PLT", "MPV", "WBC", "NEUT_ABS", "LYMP_ABS", 
            "MONO_ABS", "EOS_ABS", "BASO_ABS"
        ]
        
        # 1. تحميل الـ Tabular (scikit-learn)
        self.tabular_scaler = joblib.load(os.path.join(models_dir, "tabular_scaler.joblib"))
        self.tabular_model  = joblib.load(os.path.join(models_dir, "tabular_rf_model.joblib"))
        
        # 2. تحميل الـ Vision Model (PyTorch)
        self.vision_model = VisionOnlyResNet().to(self.device)
        self.vision_model.load_state_dict(
            torch.load(os.path.join(models_dir, "vision_only_resnet.pth"), map_location=self.device)
        )
        self.vision_model.eval() # وضع الاختبار (عشان الـ Dropout والـ BatchNorm)
        print("✅ Vision Model weights loaded successfully!")
        
        # 3. تحميل الـ Fusion Model (PyTorch)
        self.fusion_model = MultimodalFusion(tab_dim=len(self.tabular_feature_order)).to(self.device)
        self.fusion_model.load_state_dict(
            torch.load(os.path.join(models_dir, "multimodal_fusion.pth"), map_location=self.device)
        )
        self.fusion_model.eval()
        print("✅ Fusion Model weights loaded successfully!")
        
        # 4. محول الصور (Transform) زي التدريب بالظبط
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # ── Tabular preprocessing ───────────────────────────────────────────
    def preprocess_tabular(self, standardized_data: dict) -> np.ndarray:
        features = [standardized_data.get(f, 0.0) for f in self.tabular_feature_order]
        features_array = np.array(features).reshape(1, -1)
        return self.tabular_scaler.transform(features_array)

    # ── Image preprocessing ─────────────────────────────────────────────
    def preprocess_image(self, base64_str: str) -> torch.Tensor:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # تحويل الصورة لـ Tensor (3, 224, 224)
        tensor_img = self.img_transform(img)
        
        # تكرار الصورة 12 مرة عشان الموديل متدرب على 12 صورة لكل مريض
        tensor_seq = tensor_img.unsqueeze(0).repeat(12, 1, 1, 1) # (12, 3, 224, 224)
        
        # إضافة بعد الـ Batch Size (1, 12, 3, 224, 224)
        return tensor_seq.unsqueeze(0)

    # ── Prediction ──────────────────────────────────────────────────────
    def predict(self, tabular_data: dict = None, base64_image: str = None) -> Tuple[float, str]:
        if tabular_data and base64_image:
            X_tab_np = self.preprocess_tabular(tabular_data)
            X_tab = torch.tensor(X_tab_np, dtype=torch.float32).to(self.device)
            X_img = self.preprocess_image(base64_image).to(self.device)

            with torch.no_grad(): # منع حساب الـ Gradients لتوفير الميموري
                prob = self.fusion_model(X_img, X_tab).item()
            return prob, "multimodal_fusion"

        elif base64_image and not tabular_data:
            X_img = self.preprocess_image(base64_image).to(self.device)
            with torch.no_grad():
                prob = self.vision_model(X_img).item()
            return prob, "vision_only"

        elif tabular_data and not base64_image:
            X_tab_np = self.preprocess_tabular(tabular_data)
            prob = self.tabular_model.predict_proba(X_tab_np)[0][1]
            return float(prob), "tabular_only"

        else:
            raise ValueError("No data provided for inference.")


# =================================================================
# Node Class
# =================================================================
class BinaryClassifierNode:
    """
    Node 3: Determines whether the patient is sick or healthy.
    Requires the BinaryClassificationEngine passed via LangGraph config:
        config={"configurable": {"binary_engine": <engine_instance>}}
    """

    def __call__(self, state: dict, config: RunnableConfig) -> dict:
        inp   = state["input_state"]
        agent = deepcopy(state["agent_state"])  # ✅ never mutate original

        visited = list(agent.get("visited_nodes")  or [])
        trace   = list(agent.get("decision_trace") or [])
        errors  = list(agent.get("errors")         or [])

        # Mark node as visited before any try/except ─────────────────────
        visited.append("binary_classifier_node")
        agent["current_node"]  = "binary_classifier_node"
        agent["visited_nodes"] = visited

        # Safe access to RunnableConfig ───────────────────────────────────
        engine = (config or {}).get("configurable", {}).get("binary_engine")

        if not engine:
            errors.append("Binary Engine not found in config!")
            agent["errors"] = errors
            return {"agent_state": agent}

        standardized_data = agent.get("standardized_data", {})
        image_b64         = inp.get("blood_smear_image")

        has_tabular = bool(standardized_data)
        has_image   = bool(image_b64)

        try:
            prob, strategy = engine.predict(
                tabular_data = standardized_data if has_tabular else None,
                base64_image = image_b64         if has_image   else None,
            )

            is_sick = bool(prob >= 0.5)
            trace.append(f"Binary Classifier ({strategy}): Prob={prob:.4f}, is_sick={is_sick}")

            agent.update({
                "decision_trace":     trace,
                "is_sick":            is_sick,
                "disease_confidence": round(prob, 4),
            })

        except Exception as e:
            errors.append(f"Binary Classifier Error: {str(e)}")

        agent["errors"]         = errors
        agent["decision_trace"] = trace

        return {"agent_state": agent}  # ✅ return dict slice

# ──────────────────────────────────────────────
# Ready-to-use instance for LangGraph
# ──────────────────────────────────────────────
binary_classifier_node = BinaryClassifierNode()