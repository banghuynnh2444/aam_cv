"""Phase 6 - Multi-Task EfficientNet-B0 for Age Group & Gender-Label Classification.

Implements:
- MultiTaskEfficientNetB0: Shared EfficientNet-B0 backbone (224x224) with dual task heads
  for Age Group estimation and Gender-label classification (trained on FairFace).
- AttributeTrainerRTX5050: RTX 5050 8GB optimized multi-task trainer (224x224, Batch=16,
  Max Epoch=35, AMP FP16, AdamW with 1e-4 head / 1e-5 backbone, weight_decay=1e-4,
  CosineAnnealingLR, gradient clipping, early stopping, peak VRAM logging).
- Macro-F1, Balanced Accuracy, Confusion Matrix & Per-class Error Analysis utilities.
"""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Names match the FairFace age bins the model is actually trained on
# (0-2 and 3-9 merged into 0-9; 50-59, 60-69 and 70+ merged into 50+).
AGE_GROUP_CLASSES: Tuple[str, ...] = (
    "0-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50+",
)

GENDER_LABEL_CLASSES: Tuple[str, ...] = (
    "Female_Label",
    "Male_Label",
)


class MultiTaskEfficientNetB0(nn.Module):
    """Multi-Task EfficientNet-B0 for simultaneous Age Group (6 classes) and Gender-label (2 classes)."""

    def __init__(
        self,
        num_age_classes: int = len(AGE_GROUP_CLASSES),
        num_gender_classes: int = len(GENDER_LABEL_CLASSES),
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
        base = models.efficientnet_b0(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = 1280

        self.age_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_age_classes),
        )
        self.gender_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_gender_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        pooled = torch.flatten(self.avgpool(feat), 1)
        age_logits = self.age_head(pooled)
        gender_logits = self.gender_head(pooled)
        return age_logits, gender_logits


class AttributeTrainerRTX5050:
    """RTX 5050 8GB Multi-Task Trainer for EfficientNet-B0 (Batch=16, Max Epoch=35, AMP FP16)."""

    def __init__(
        self,
        model: MultiTaskEfficientNetB0,
        device: Optional[str] = None,
        lr_head: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        max_epochs: int = 35,
        grad_clip: float = 1.0,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.max_epochs = int(max_epochs)
        self.grad_clip = float(grad_clip)
        self.use_amp = self.device.type == "cuda"

        head_params = list(self.model.age_head.parameters()) + list(self.model.gender_head.parameters())
        backbone_params = list(self.model.features.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": lr_head},
                {"params": backbone_params, "lr": lr_backbone},
            ],
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.max_epochs), eta_min=1e-6
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_step(
        self,
        images: torch.Tensor,
        age_labels: torch.Tensor,
        gender_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        age_labels = age_labels.to(self.device, non_blocking=True)
        gender_labels = gender_labels.to(self.device, non_blocking=True)

        if self.use_amp:
            torch.cuda.reset_peak_memory_stats(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            age_logits, gender_logits = self.model(images)
            loss_age = F.cross_entropy(age_logits, age_labels)
            loss_gender = F.cross_entropy(gender_logits, gender_labels)
            total_loss = loss_age + loss_gender

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(self.device)) / (1024 ** 2)
            if self.use_amp
            else 0.0
        )
        return {
            "total_loss": round(float(total_loss.item()), 4),
            "age_loss": round(float(loss_age.item()), 4),
            "gender_loss": round(float(loss_gender.item()), 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Tuple[str, ...],
) -> Dict[str, Any]:
    """Computes Macro-F1, Balanced Accuracy, Confusion Matrix, and Per-Class Error Analysis."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    num_classes = len(class_names)

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    f1_list = []
    recall_list = []
    per_class = {}

    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        support = tp + fn

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, support)
        f1 = (2.0 * prec * rec) / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0

        if support > 0:
            f1_list.append(f1)
            recall_list.append(rec)

        per_class[class_names[c]] = {
            "support": support,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "error_rate": round(float(1.0 - rec) if support > 0 else 0.0, 4),
        }

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    balanced_acc = float(np.mean(recall_list)) if recall_list else 0.0
    overall_acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0

    return {
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_analysis": per_class,
    }
