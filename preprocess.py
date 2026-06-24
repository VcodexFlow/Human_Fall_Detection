import os
import glob
import argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Preprocessing for ST-GCN Slip and Fall Detection")
    parser.add_argument("--dataset_type", type=str, default="generic", choices=["generic", "urfall", "upfall", "le2i"],
                        help="Dataset type to process. Generic uses folder-based classification (falls vs adls).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the dataset root folder.")
    parser.add_argument("--output_dir", type=str, default="data_processed",
                        help="Directory to save the processed .npy split files.")
    parser.add_argument("--seq_len", type=str, default="50",
                        help="Number of frames per sequence.")
    parser.add_argument("--overlap", type=str, default="25",
                        help="Frame overlap for sliding window extraction (only for fall sequences/generic).")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-pose.pt",
                        help="YOLOv8 pose model weights.")
    return parser.parse_args()

def normalize_keypoints(kpts):
    """
    Applies center-of-gravity translation normalization and scale invariance
    while preserving the anatomical skeleton aspect ratio.
    kpts: numpy array of shape (17, 3) where columns are (x, y, confidence)
    """
    normalized = np.zeros_like(kpts)
    x = kpts[:, 0]
    y = kpts[:, 1]
    scores = kpts[:, 2]
    
    valid_mask = scores > 0.1
    if valid_mask.any():
        # 1. Translate center of gravity to origin (0, 0)
        center_x = x[valid_mask].mean()
        center_y = y[valid_mask].mean()
        x_centered = x - center_x
        y_centered = y - center_y
        
        # 2. Scale by the maximum distance from center of gravity
        dists = np.sqrt(x_centered[valid_mask]**2 + y_centered[valid_mask]**2)
        scale = dists.max() if len(dists) > 0 else 1.0
        scale = max(scale, 1e-5)
        
        normalized[:, 0] = x_centered / scale
        normalized[:, 1] = y_centered / scale
        normalized[:, 2] = scores
        
        # Zero out invalid points
        normalized[~valid_mask, 0] = 0
        normalized[~valid_mask, 1] = 0
    return normalized

def extract_keypoints_from_video(video_path, yolo_model, device):
    """
    Extracts, normalizes, and tracks keypoints frame-by-frame from a video.
    Returns: numpy array of shape (num_frames, 3, 17, 1)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
        
    frames_kpts = []
    last_valid_kpts = np.zeros((17, 3))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLOv8-pose inference on the frame
        results = yolo_model(frame, verbose=False, device=device, classes=0, conf=0.35)
        result = results[0]
        
        found_person = False
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                # Double check class ID (0 is person)
                if result.boxes[i].cls is not None:
                    cls_val = result.boxes[i].cls
                    class_id = int(cls_val[0].item()) if len(cls_val) > 0 else 0
                    if class_id != 0:
                        continue
                
                if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints.data) > i:
                    person_kpts = result.keypoints.data[i].cpu().numpy()
                    valid_kpts_count = np.sum(person_kpts[:, 2] > 0.35)
                    if valid_kpts_count >= 5:
                        normalized_kpts = normalize_keypoints(person_kpts)
                        last_valid_kpts = normalized_kpts.copy()
                        frames_kpts.append(normalized_kpts)
                        found_person = True
                        break
                        
        if not found_person:
            # Use last valid keypoints if tracking is temporarily lost
            frames_kpts.append(last_valid_kpts.copy())
            
    cap.release()
    
    if len(frames_kpts) == 0:
        return None
        
    # Reshape to (T, C, V, M) where C=3, V=17, M=1
    # frames_kpts: List of shape (T, 17, 3)
    kpts_array = np.array(frames_kpts)  # (T, 17, 3)
    kpts_array = np.transpose(kpts_array, (0, 2, 1))  # (T, 3, 17)
    kpts_array = np.expand_dims(kpts_array, axis=-1)  # (T, 3, 17, 1)
    return kpts_array

def segment_sequence(kpts_video, seq_len, overlap):
    """
    Uses a sliding window to segment variable-length videos into fixed size seq_len arrays.
    kpts_video: numpy array of shape (T, 3, 17, 1)
    Returns: list of arrays of shape (3, seq_len, 17, 1)
    """
    num_frames = kpts_video.shape[0]
    sequences = []
    
    if num_frames < seq_len:
        # Pad shorter videos by repeating the last frame
        pad_size = seq_len - num_frames
        last_frame = kpts_video[-1:]
        padding = np.repeat(last_frame, pad_size, axis=0)
        padded = np.concatenate([kpts_video, padding], axis=0)  # (seq_len, 3, 17, 1)
        # Transpose to shape (3, seq_len, 17, 1)
        sequences.append(np.transpose(padded, (1, 0, 2, 3)))
    else:
        # Slide window
        step = seq_len - overlap
        for i in range(0, num_frames - seq_len + 1, step):
            segment = kpts_video[i : i + seq_len]
            sequences.append(np.transpose(segment, (1, 0, 2, 3)))
            
        # Make sure we capture the end of the video if there's remaining frames
        if (num_frames - seq_len) % step != 0:
            segment = kpts_video[-seq_len:]
            sequences.append(np.transpose(segment, (1, 0, 2, 3)))
            
    return sequences

def get_label_from_path(file_path, dataset_type):
    """
    Identifies binary label (1: Fall, 0: Non-Fall/ADL) based on filename or parent directory.
    """
    filename = os.path.basename(file_path).lower()
    parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
    
    if dataset_type == "urfall":
        # UR-Fall: files containing 'fall' are falls, 'adl' are non-falls
        if "fall" in filename and "adl" not in filename:
            return 1
        return 0
        
    elif dataset_type == "upfall":
        # UP-Fall format: "SubjectX_ActivityY_TrialZ.avi"
        # Activity 1 to 5: Falls (label 1)
        # Activity 6 to 11: ADLs (label 0)
        try:
            parts = filename.split('_')
            for part in parts:
                if "activity" in part:
                    act_val = int(''.join(filter(str.isdigit, part)))
                    if 1 <= act_val <= 5:
                        return 1
                    return 0
        except Exception:
            pass
        # Fallback to simple folder/name checks
        if "fall" in filename or "activity1" in filename or "activity2" in filename or "activity3" in filename or "activity4" in filename or "activity5" in filename:
            return 1
        return 0
        
    elif dataset_type == "le2i":
        # Le2i format: subfolders containing 'fall' are falls, others are normal
        if "fall" in parent_dir or "fall" in filename:
            return 1
        return 0
        
    else:  # generic
        # Generic folder/file prefix structure
        if "fall" in parent_dir or "fall" in filename:
            return 1
        return 0

def main():
    args = parse_args()
    seq_len = int(args.seq_len)
    overlap = int(args.overlap)
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA) is not reachable or not available, but forced GPU execution was requested for preprocessing!")
    device = "cuda"
    print(f"Preprocessing running on device: {device}")
    
    # Load YOLOv8 Pose model
    yolo_model = YOLO(args.yolo_model)
    
    # Find all video files recursively (.mp4, .avi, .mkv)
    video_extensions = ["*.mp4", "*.avi", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
        
    print(f"Found {len(video_files)} video clips to process in {args.input_dir}")
    
    sequences_all = []
    labels_all = []
    
    for idx, video_path in enumerate(video_files):
        label = get_label_from_path(video_path, args.dataset_type)
        print(f"[{idx+1}/{len(video_files)}] Processing {os.path.basename(video_path)} (Label: {'Fall' if label == 1 else 'Non-Fall'})...")
        
        # 1. Extract skeleton coordinates
        kpts_video = extract_keypoints_from_video(video_path, yolo_model, device)
        if kpts_video is None:
            print(f"  Warning: No skeletons extracted for {os.path.basename(video_path)}. Skipping.")
            continue
            
        # 2. Slice into sequences of length seq_len
        segmented = segment_sequence(kpts_video, seq_len, overlap)
        
        sequences_all.extend(segmented)
        labels_all.extend([label] * len(segmented))
        
    if len(sequences_all) == 0:
        print("Error: No valid data sequences could be processed. Please check input directories.")
        return
        
    # Convert lists to NumPy arrays
    X = np.array(sequences_all, dtype=np.float32)  # (N, 3, seq_len, 17, 1)
    y = np.array(labels_all, dtype=np.int64)      # (N,)
    
    print(f"Processed dataset shape: {X.shape}, labels: {y.shape}")
    print(f"Class distribution: Falls: {np.sum(y == 1)}, Non-Falls: {np.sum(y == 0)}")
    
    # Generate splits: 70% Train, 15% Val, 15% Test
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    except Exception:
        print("  Warning: Dataset is too small or unbalanced for stratified split. Falling back to simple random split.")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    # Save splits
    os.makedirs(args.output_dir, exist_ok=True)
    
    np.save(os.path.join(args.output_dir, "x_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "x_val.npy"), X_val)
    np.save(os.path.join(args.output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.output_dir, "x_test.npy"), X_test)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)
    
    print(f"Data saved successfully to {args.output_dir}:")
    print(f"  Train: {X_train.shape} | Labels: {y_train.shape}")
    print(f"  Val:   {X_val.shape} | Labels: {y_val.shape}")
    print(f"  Test:  {X_test.shape} | Labels: {y_test.shape}")

if __name__ == "__main__":
    main()
