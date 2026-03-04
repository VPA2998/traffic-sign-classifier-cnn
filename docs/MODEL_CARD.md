# 🤖 Model Card: Traffic Sign Classifier (CNN + CLIP)

## 📋 Model Details

- **Type:** Convolutional Neural Network (CNN) + CLIP for explanations
- **Architecture:** 2 Conv layers + 2 FC layers (~549K parameters)
- **Input:** 32x32 RGB images
- **Output:** 43-class probability distribution (GTSRB classes)
- **Framework:** PyTorch 2.0+

---

## 🎯 Intended Use

**Primary Use Cases:**
- Traffic sign recognition for ADAS prototypes
- Educational demonstrations of computer vision
- Research on model robustness to image corruptions

**Out-of-Scope Uses:**
- Safety-critical deployment without additional validation
- Night driving or extreme weather conditions
- Recognition of non-GTSRB traffic signs (e.g., US-specific signs)

---

## 📊 Training Data

**Dataset:** GTSRB (German Traffic Sign Recognition Benchmark) via Hugging Face (`tanganke/gtsrb`)

| Split | Samples | Source |
|-------|---------|--------|
| **Training** | 10,000 | Selected from original train set |
| **Test** | 1,000 | Selected from original test set |

**Classes:** 43 German traffic sign categories (speed limits, warnings, mandatory signs)

**Augmentation:** Tested robustness on 7 corruption types (contrast, noise, blur, JPEG, pixelation, spatter).

---

## 📈 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | **93.4%** | On 1,000 held-out samples |
| **Training Accuracy** | 92.1% | After 15 epochs |
| **Inference Time** | ~2ms | On CPU (single image) |
| **Model Size** | ~2.2 MB | Saved weights (.pth) |

**Limitations:**
- Performance drops on heavily occluded signs
- Not evaluated on real-world dashcam footage
- CLIP explanations are descriptive, not diagnostic

---

## ⚠️ Ethical Considerations

**Bias:**
- Trained only on German traffic signs; may not generalize to other regions
- Dataset balanced across classes, but real-world distribution may vary

**Safety:**
- **NOT** certified for autonomous driving or safety-critical systems
- Requires human oversight in any deployment scenario
- Should be validated on target hardware and environment before use

**Privacy:**
- No personal data collected or processed
- GTSRB dataset is publicly available and anonymized

---

## 🔧 Recommendations

**For Researchers:**
- Fine-tune on region-specific datasets for local deployment
- Explore knowledge distillation for edge deployment
- Investigate adversarial robustness

**For Engineers:**
- Add runtime sanity checks (e.g., confidence thresholds)
- Combine with other sensors (LiDAR, radar) for redundancy
- Monitor prediction drift in production

---

## 📚 Citations

- **Dataset:** Stallkamp et al., "The German Traffic Sign Recognition Benchmark", IJCNN 2011.
- **CLIP Model:** Radford et al., "Learning Transferable Visual Models from Natural Language Supervision", ICML 2021.

---
