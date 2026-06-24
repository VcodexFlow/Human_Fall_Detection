import os
import argparse
import torch
import numpy as np
import cv2
from ultralytics import YOLO

from model.st_gcn import STGCN
from preprocess import normalize_keypoints, extract_keypoints_from_video, segment_sequence

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for ST-GCN Slip and Fall Detection")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to raw video file (.mp4, .avi, etc.) OR a pre-extracted keypoint file (.npy).")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained ST-GCN model weights checkpoint.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-pose.pt",
                        help="YOLOv8 pose model weights (only used if source is a video).")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="Target sequence length (number of frames).")
    parser.add_argument("--overlap", type=int, default=25,
                        help="Overlap between sliding windows for video segmentation.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cpu or cuda).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Select Device
    if args.device:
        device = torch.device(args.device)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU (CUDA) is not reachable or not available, but forced GPU execution was requested for inference!")
        device = torch.device("cuda")
        
    print(f"Running inference on device: {device}")
    
    # 2. Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: ST-GCN model weights not found at: {args.model_path}")
        print("Please train a model first or check your path.")
        return
        
    # 3. Instantiate and Load ST-GCN Model
    model = STGCN(in_channels=3, num_classes=2).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("ST-GCN model loaded successfully.")
    
    # 4. Handle Input
    source_ext = os.path.splitext(args.source)[1].lower()
    
    if source_ext == ".npy":
        # Load pre-extracted keypoint file
        print(f"Loading keypoints from: {args.source}")
        kpts = np.load(args.source)
        
        # Adjust dimensions to match (N, C, T, V, M)
        # Expected input shape of a single sequence: (3, seq_len, 17, 1)
        if len(kpts.shape) == 4:
            # (3, T, V, 1) -> Add batch dimension
            inputs = np.expand_dims(kpts, axis=0)
        elif len(kpts.shape) == 5:
            # Already (N, 3, T, V, M)
            inputs = kpts
        else:
            print(f"Error: Invalid NumPy array shape {kpts.shape}. Expected 4D or 5D array.")
            return
            
        print(f"Input tensor shape formatted for ST-GCN: {inputs.shape}")
        
        with torch.no_grad():
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
            logits = model(inputs_tensor)
            probs = torch.softmax(logits, dim=1)
            
        for i in range(len(inputs)):
            prob_fall = probs[i, 1].item()
            pred_class = "Fall" if prob_fall > 0.5 else "Non-Fall"
            print(f"Sample {i:d} | Prediction: {pred_class:<8} | Fall Probability: {prob_fall*100:6.2f}%")
            
    else:
        # Load raw video file
        print(f"Processing raw video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.source}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # VideoWriter disabled per user request (no local saves)
        
        print("Loading YOLOv8-pose model...")
        yolo_model = YOLO(args.yolo_model)
        
        # Dictionary to store tracking state for each person
        # Maps track_id -> dict containing queues, last valid values, and metrics
        people_tracks = {}
        
        print("Extracting poses, running ST-GCN, and displaying pop-up preview...")
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLOv8-pose tracking on the frame
            results = yolo_model.track(frame, persist=True, verbose=False, device=device.type, classes=0, conf=0.35)
            result = results[0]
            
            # Identify active track IDs in this frame
            active_ids_this_frame = set()
            
            # Determine valid person detections based on boxes, class, and keypoints
            valid_person_indices = []
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                # 1. Gather all class 0 (person) detections
                detections = []
                for i in range(len(result.boxes)):
                    # Check class ID (0 is person)
                    if result.boxes[i].cls is not None:
                        cls_val = result.boxes[i].cls
                        class_id = int(cls_val[0].item()) if len(cls_val) > 0 else 0
                        if class_id != 0:
                            continue
                            
                    bbox = result.boxes[i].xyxy[0].cpu().numpy()
                    conf_score = float(result.boxes[i].conf[0].item()) if result.boxes[i].conf is not None else 0.5
                    detections.append({
                        'index': i,
                        'bbox': bbox,
                        'conf': conf_score,
                        'suppressed': False
                    })
                
                # 2. Suppress overlapping duplicate boxes
                for i in range(len(detections)):
                    if detections[i]['suppressed']:
                        continue
                    for j in range(i + 1, len(detections)):
                        if detections[j]['suppressed']:
                            continue
                        
                        # Compute IoU and overlap ratio
                        box1 = detections[i]['bbox']
                        box2 = detections[j]['bbox']
                        
                        x1 = max(box1[0], box2[0])
                        y1 = max(box1[1], box2[1])
                        x2 = min(box1[2], box2[2])
                        y2 = min(box1[3], box2[3])
                        
                        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        union_area = area1 + area2 - inter_area
                        
                        iou = inter_area / union_area if union_area > 0 else 0.0
                        min_area = min(area1, area2)
                        overlap_ratio = inter_area / min_area if min_area > 0 else 0.0
                        
                        # If boxes overlap significantly, suppress the one with lower confidence
                        if iou > 0.45 or overlap_ratio > 0.70:
                            if detections[i]['conf'] >= detections[j]['conf']:
                                detections[j]['suppressed'] = True
                            else:
                                detections[i]['suppressed'] = True
                                break  # Outer box is suppressed, stop checking against it
                                
                # Keep only non-suppressed detections
                valid_person_indices = [d['index'] for d in detections if not d['suppressed']]
            
            if len(valid_person_indices) > 0:
                # Fallback to unique temporary IDs if tracking ID is not initialized yet
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                else:
                    track_ids = list(range(1000, 1000 + len(result.boxes)))
                
                for i in valid_person_indices:
                    track_id = track_ids[i]
                    active_ids_this_frame.add(track_id)
                    
                    if track_id not in people_tracks:
                        people_tracks[track_id] = {
                            'kpts_queue': [],
                            'raw_kpts_queue': [],
                            'aspect_ratio_queue': [],
                            'last_valid_kpts': np.zeros((17, 3)),
                            'last_valid_raw_kpts': np.zeros((17, 3)),
                            'last_valid_bbox': None,
                            'max_fall_prob': 0.0,
                            'current_bbox': None,
                            'fall_prob': 0.0
                        }
                    
                    track_data = people_tracks[track_id]
                    
                    # Extract bbox and keypoints for this person
                    bbox = result.boxes[i].xyxy[0].cpu().numpy()
                    track_data['last_valid_bbox'] = bbox.copy()
                    track_data['current_bbox'] = bbox.copy()
                    
                    # Verify keypoints confidence before running ST-GCN or updating queues
                    has_valid_kpts = False
                    if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints.data) > i:
                        person_kpts = result.keypoints.data[i].cpu().numpy()
                        valid_kpts_count = np.sum(person_kpts[:, 2] > 0.35)
                        if valid_kpts_count >= 5:
                            has_valid_kpts = True
                            
                    # Update keypoint queues and run ST-GCN only if keypoints are valid
                    if has_valid_kpts:
                        normalized_kpts = normalize_keypoints(person_kpts)
                        track_data['last_valid_kpts'] = normalized_kpts.copy()
                        track_data['last_valid_raw_kpts'] = person_kpts.copy()
                        
                        # Add keypoints to queues
                        track_data['kpts_queue'].append(normalized_kpts)
                        track_data['raw_kpts_queue'].append(person_kpts)
                        
                        # Calculate aspect ratio of bounding box
                        w_box = max(bbox[2] - bbox[0], 1.0)
                        h_box = max(bbox[3] - bbox[1], 1.0)
                        ratio = h_box / w_box
                        track_data['aspect_ratio_queue'].append(ratio)
                        
                        # Maintain queue lengths of size seq_len
                        if len(track_data['kpts_queue']) > args.seq_len:
                            track_data['kpts_queue'].pop(0)
                        if len(track_data['raw_kpts_queue']) > args.seq_len:
                            track_data['raw_kpts_queue'].pop(0)
                        if len(track_data['aspect_ratio_queue']) > args.seq_len:
                            track_data['aspect_ratio_queue'].pop(0)
                            
                        # Run ST-GCN sequence model on their rolling sequence
                        fall_prob = 0.0
                        k_queue = track_data['kpts_queue']
                        raw_k_queue = track_data['raw_kpts_queue']
                        ratio_queue = track_data['aspect_ratio_queue']
                        
                        if len(k_queue) > 0:
                            if len(k_queue) == args.seq_len:
                                kpts_arr = np.array(k_queue)  # (seq_len, 17, 3)
                            else:
                                # Pad sequence to target seq_len by repeating the last frame
                                current_len = len(k_queue)
                                pad_size = args.seq_len - current_len
                                last_item = k_queue[-1]
                                padding = [last_item] * pad_size
                                kpts_arr = np.array(k_queue + padding)  # (seq_len, 17, 3)
                            
                            # Format to shape (1, 3, seq_len, 17, 1)
                            kpts_arr = np.transpose(kpts_arr, (2, 0, 1))  # (3, seq_len, 17)
                            kpts_arr = np.expand_dims(kpts_arr, axis=0)  # (1, 3, seq_len, 17)
                            kpts_arr = np.expand_dims(kpts_arr, axis=-1)  # (1, 3, seq_len, 17, 1)
                            
                            with torch.no_grad():
                                inputs_tensor = torch.tensor(kpts_arr, dtype=torch.float32).to(device)
                                logits = model(inputs_tensor)
                                probs = torch.softmax(logits, dim=1)
                                fall_prob = probs[0, 1].item()
                                
                            # Apply motion constraint to suppress false positives for static sleeping/resting postures
                            if len(raw_k_queue) >= 5:
                                valid_raw_kpts = [k for k in raw_k_queue if np.any(k)]
                                if len(valid_raw_kpts) >= 5:
                                    raw_kpts_arr = np.array(valid_raw_kpts)  # (L, 17, 3)
                                    y_coords = raw_kpts_arr[:, :, 1]  # (L, 17)
                                    y_std = np.std(y_coords, axis=0)  # (17,)
                                    avg_y_std = np.mean(y_std)
                                    
                                    body_scale = max(w_box, h_box)
                                    norm_motion = avg_y_std / body_scale
                                    
                                    # If vertical motion is extremely low, scale down the fall probability
                                    motion_threshold = 0.05
                                    if norm_motion < motion_threshold:
                                        fall_prob = fall_prob * (norm_motion / motion_threshold)
                                        
                            # Apply aspect ratio history check: a fall transition requires standing/sitting history (aspect ratio > 1.1)
                            # Relax for very short sequences (less than 80 frames) where the fall may start immediately
                            max_ratio_in_window = max(ratio_queue) if len(ratio_queue) > 0 else 1.0
                            required_ratio = 1.1 if total_frames > 80 else 0.8
                            if max_ratio_in_window < required_ratio:
                                fall_prob = 0.0
                                
                        track_data['fall_prob'] = fall_prob
                        if fall_prob > track_data['max_fall_prob']:
                            track_data['max_fall_prob'] = fall_prob
                            
            # Set current_bbox to None for any track ID that was not active in this frame
            for track_id, track_data in people_tracks.items():
                if track_id not in active_ids_this_frame:
                    track_data['current_bbox'] = None
            
            # Draw bounding boxes and text overlays on frame for all active detections
            if len(valid_person_indices) > 0:
                for i in valid_person_indices:
                    track_id = track_ids[i]
                    if track_id in people_tracks:
                        track_data = people_tracks[track_id]
                        current_bbox = track_data['current_bbox']
                        fall_prob = track_data['fall_prob']
                        
                        if current_bbox is not None:
                            xmin, ymin, xmax, ymax = map(int, current_bbox)
                            
                            # Visual thresholding: turns RED if fall probability > 50%
                            is_fall = fall_prob > 0.5
                            color = (0, 0, 255) if is_fall else (0, 255, 0)  # BGR: Red / Green
                            label_text = f"ID {track_id} FALL: {fall_prob*100:.1f}%" if is_fall else f"ID {track_id} Normal: {fall_prob*100:.1f}%"
                            
                            # Draw box
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                            # Label box background
                            cv2.rectangle(frame, (xmin, max(0, ymin - 30)), (xmin + len(label_text)*13, ymin), color, -1)
                            # Label text
                            cv2.putText(frame, label_text, (xmin + 5, max(15, ymin - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Drawing overlays directly on the display frame (no save to local disk)
            
            # Display the frame in a pop-up window (resized to a smaller width of 640px)
            max_disp_width = 640
            if width > max_disp_width:
                scale_ratio = max_disp_width / width
                disp_w = int(width * scale_ratio)
                disp_h = int(height * scale_ratio)
                display_frame = cv2.resize(frame, (disp_w, disp_h))
            else:
                display_frame = frame
                
            cv2.imshow("ST-GCN Fall Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Inference interrupted by user.")
                break
                
            frame_idx += 1
            
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("                 DETECTION SUMMARY")
        print("="*60)
        print(f"Total Frames Processed: {frame_idx}")
        
        any_fall = False
        for track_id, track_data in people_tracks.items():
            max_prob = track_data['max_fall_prob']
            status = "FALL" if max_prob > 0.5 else "NON-FALL"
            print(f"Person ID {track_id:<3} | Max Fall Probability: {max_prob*100:6.2f}% | Status: {status}")
            if max_prob > 0.5:
                any_fall = True
                
        final_class = "FALL" if any_fall else "NON-FALL"
        print(f"Overall Video Classification: {final_class}")
        print("Real-time inference completed successfully.")
        print("="*60)

if __name__ == "__main__":
    main()
