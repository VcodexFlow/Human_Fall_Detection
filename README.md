# Spatial Temporal Graph Convolutional Network (ST-GCN) for Slip and Fall Detection

This repository implements a production-quality, modular **Spatial Temporal Graph Convolutional Network (ST-GCN)** in PyTorch for classifying human movement sequences as **Fall** or **Non-Fall** based on human skeleton keypoints.

Optimized for **NVIDIA RTX 3060 6GB** and **CUDA**, this implementation extracts keypoints dynamically from raw video files using YOLOv8-pose, normalizes coordinates to ensure scale and translation invariance, segments temporal streams using a sliding window, and classifies movements using spatio-temporal graphs.

---

## Repository Structure

```
├── requirements.txt         # Package dependencies
├── README.md                # System documentation
├── model/
│   ├── __init__.py          # Model package index
│   ├── graph.py             # Skeleton adjacency graph (COCO-17 Layout)
│   └── st_gcn.py            # ST-GCN architecture layers and classifier
├── preprocess.py            # Dataset pose-extraction and normalization pipeline
├── dataset.py               # Skeletal Dataset loader with data augmentations
├── train.py                 # Training script with AMP, early stopping, and TensorBoard logs
├── evaluate.py              # Test set evaluation script (Accuracy, Recall, ROC-AUC)
└── inference.py             # Raw Video / Pose sequence prediction tool
```

---

## Features & Optimizations

- **ST-GCN Network Architecture**: Implements Yan et al. (AAAI 2018) Spatial Temporal Graph Convolutions with learnable edge importance weights to dynamically weight skeletal joints.
- **Advanced Skeleton Normalization**: Centered at the skeleton center-of-gravity and scaled symmetrically to preserve the anatomical aspect ratio while providing scale and location invariance.
- **Skeletal Data Augmentation**: Real-time Gaussian joint coordinate jittering, random 2D rotation, scaling, and joint masking (dropout) to handle occlusions.
- **Automatic Mixed Precision (AMP)**: Uses `torch.cuda.amp` to accelerate training and reduce memory utilization on GPU (NVIDIA RTX 3060 6GB).
- **Flexible Data Input**: Preprocessing supports **UR-Fall**, **UP-Fall**, **Le2i**, and a **Generic folder structure**.
- **Sliding-Window Inference**: Processes videos of arbitrary length, producing a timestamp-based prediction timeline showing probability for each interval.

---

## Installation

Ensure you have PyTorch with CUDA support enabled, then install the package requirements:

```bash
# Activate your virtual environment and run:
pip install -r requirements.txt
```

### 📥 Model Weights Configuration

To run evaluation or inference, you need the pre-trained ST-GCN model weights:
1. Create a `checkpoints` directory in the project root:
   ```bash
   mkdir checkpoints
   ```
2. Download the pre-trained model weights `best_model.pt` from the **Releases** page of this repository and place them inside the `checkpoints/` folder.

---

## Dataset Processing (`preprocess.py`)

The preprocessing script searches for video files, extracts 17 COCO joints using YOLOv8-pose, applies normalization, segments frames, and saves processed splits into train/val/test `.npy` files.

### Dataset Folders Organization

#### 1. Generic Folder Layout
Store files under `falls` and `adls` (or any other folder names; files or parents containing "fall" are treated as positive):
```
dataset_root/
├── falls/
│   ├── fall_01.mp4
│   └── fall_02.avi
└── adls/
    ├── walking_01.mp4
    └── sitting_01.avi
```
Run preprocessing:
```bash
python preprocess.py --dataset_type generic --input_dir path/to/dataset_root --output_dir data_processed
```

#### 2. UR-Fall Dataset
Filename conventions must follow `fall-XX-cam0.mp4` for falls and `adl-XX-cam0.mp4` for daily activities.
Run:
```bash
python preprocess.py --dataset_type urfall --input_dir path/to/urfall_root --output_dir data_processed
```

#### 3. UP-Fall Dataset
Filename conventions should correspond to `SubjectX_ActivityY_TrialZ.avi`. Activity codes 1-5 represent falls, 6-11 represent ADLs.
Run:
```bash
python preprocess.py --dataset_type upfall --input_dir path/to/upfall_root --output_dir data_processed
```

#### 4. Le2i Dataset
Le2i videos are sorted under parent directories representing locations (Home, Office, Coffee_room) containing folders named `Fall` and `Normal`.
Run:
```bash
python preprocess.py --dataset_type le2i --input_dir path/to/le2i_root --output_dir data_processed
```

---

## Training (`train.py`)

Train the model on the preprocessed dataset. Supported configurations include learning-rate decay, TensorBoard logs, checkpointing, and early stopping.

```bash
python train.py --data_dir data_processed --epochs 60 --batch_size 32 --lr 0.001 --mixed_precision
```

### Verification (Mock Mode)
To test and verify the entire training pipeline without downloading raw datasets first, run the training pipeline in **Mock Mode**:
```bash
python train.py --mock --epochs 5 --batch_size 16
```
This automatically generates a synthetic dataset in `data_processed/` and runs training, validating model execution, mixed-precision CUDA acceleration, checkpointing, and logging.

### Monitoring with TensorBoard
View real-time training and validation losses/accuracies in the browser:
```bash
tensorboard --logdir runs
```

---

## Evaluation (`evaluate.py`)

Test the model weights against the test split, printing summary statistics and saving ROC Curves and Confusion Matrices.

```bash
python evaluate.py --data_dir data_processed --model_path checkpoints/best_model.pt --output_dir evaluation_results
```
Outputs saved in `evaluation_results/`:
- `report.txt`: Final metrics summary (Accuracy, Precision, Recall, F1, ROC-AUC).
- `confusion_matrix.png`: Plotted confusion matrix.
- `roc_curve.png`: Plotted ROC curve.

---

## Inference (`inference.py`)

Run inference on a raw video file or a preprocessed `.npy` file. For videos, it classifies each sliding window segment, outputting an action timeline:

### 1. Run on a Video File
```bash
python inference.py --source path/to/test_video.mp4 --model_path checkpoints/best_model.pt
```

### 2. Run on a Keypoint Sequence
```bash
python inference.py --source data_processed/x_test.npy --model_path checkpoints/best_model.pt
```

### Timeline Output Example
When processing a raw video, the model segments the frame sequence and identifies the specific intervals containing falls:
```
============================================================
                 DETECTION TIMELINE
============================================================
✅ Seg 00 ( 0.00s -  1.67s) | Result: Normal Activity | Fall Probability:  0.12%
✅ Seg 01 ( 0.83s -  2.50s) | Result: Normal Activity | Fall Probability:  0.89%
⚠️ Seg 02 ( 1.67s -  3.33s) | Result: FALL DETECTED!  | Fall Probability: 78.43%
⚠️ Seg 03 ( 2.50s -  4.17s) | Result: FALL DETECTED!  | Fall Probability: 99.85%
============================================================
Overall Classification: FALL
Maximum Fall Probability: 99.85%
============================================================
```
