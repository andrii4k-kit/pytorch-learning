# PyTorch Learning Journey & Defect Detection Project✨

This repository documents an intensive 6-week technical onboarding process for my position as a Student Research Assistant (HiWi) at the Karlsruhe Institute of Technology (KIT).

## 🎯 Final Project: Surface Defect Detection
I applied the learned concepts to build a Quality Control pipeline for magnetic tiles.

* **Model:** RT-DETR (Real-Time Detection Transformer)
* **Features:** Hybrid approach using Object Detection + SAM2 for Segmentation.
* **Data Engineering:** Custom pipeline to convert pixel masks to bounding boxes (see `07_data_preparation.ipynb`).
<img width="1229" height="1198" alt="output" src="https://github.com/user-attachments/assets/2d447f47-81d8-40bc-b118-7751f3f80da2" />

---

## 📚 The Learning Path (Notebooks)
I built this knowledge base following the "Zero to Mastery" curriculum.

| File | Topic | Key Learnings |
|------|-------|---------------|
| `00_fundamentals.ipynb` | **Tensors** | GPU/MPS usage, Matrix multiplication, Tensor shapes |
| `01_workflow.ipynb` | **Workflow** | Training loops, Loss functions, Optimizers (SGD/Adam) |
| `02_nn_classification.ipynb`| **Neural Nets** | Non-linearity (ReLU/Sigmoid), Multi-class classification |
| `03_Computer_Vision_CNN.ipynb` | **CNNs** | Conv2d, MaxPool2d, recreating TinyVGG architecture |
| `04_custom_datasets_f.ipynb` | **Data Loading** | Writing custom `Dataset` classes, Transforms, Augmentation |
| `05_modularity_PyTorchh.ipynb` | **Software Eng.** | Refactoring notebooks into Python scripts (`data_setup.py`, `engine.py`) |
| `06_transfer_learning.ipynb` | **Transfer Learning** | Fine-tuning EfficientNet/ResNet for custom tasks |

## 📉 Challenges & Retrospective
During the project, I encountered specific data challenges that provided valuable insights for future iterations:

### 1. Dataset Imbalance
Source: [Magnetic Tile Surface Defects (Kaggle)](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects)
The dataset contains ~1,300 images, but is heavily skewed:
* **Background (No Defect):** ~850 images (crucial for reducing false positives).
* **Defects (5 Classes):** Only ~90 images per class.
* *Impact:* The model is "low-shot" learning on defects, making it sensitive to lighting changes.

### 2. Preprocessing Artifacts (Mask-to-Box Conversion)
I implemented a custom algorithm to convert segmentation masks into YOLO bounding boxes (see `07_data_preparation.ipynb`). 
* **Issue Identified:** Post-training analysis revealed that large, disconnected defects (e.g., complex cracks) were sometimes segmented into multiple small boxes instead of one enclosing box.
* **Result:** The model occasionally predicts multiple small bounding boxes for a single large defect.
* **Lesson Learned:** Future improvements would require a morphological operation (e.g., dilation) to merge nearby pixel components before generating the bounding box.


## ⚙️ Training Configuration
I trained the model on Kaggle (using T4 GPUs) for **300 epochs** with an aggressive augmentation strategy designed to simulate industrial environments (lighting changes, rotation).

### Key Hyperparameters & Augmentations
```python
model.train(
    data='magnetic_tiles.yaml',
    epochs=300,
    imgsz=640,
    batch=32,
    
    # --- Industrial Augmentation Strategy ---
    degrees=90.0,      # +/- 90° rotation (Crucial: defect type is rotation-invariant)
    flipud=0.6,        # Vertical flip (Top-down view symmetry)
    fliplr=0.6,        # Horizontal flip
    mosaic=1.0,        # Strong mosaic to detect small defects relative to tile size
    mixup=0.3,         # Improves robustness against noise
    
    # --- Lighting Simulation ---
    hsv_h=0.03, hsv_s=0.7, hsv_v=0.6  # Heavy color/brightness jitter to handle factory lighting
)
```
<img width="2400" height="1200" alt="many_graphs" src="https://github.com/user-attachments/assets/a151b45e-e544-4f8c-8c74-95d81e6bce00" />
## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Core Libs:** PyTorch, Torchvision, Ultralytics (YOLO/RT-DETR), OpenVINO
* **Tools:** Jupyter, Matplotlib, Scikit-Image

## 🚧 Status
* [x] Fundamentals & Theory
* [x] Software Prototype (Laptop Inference)
