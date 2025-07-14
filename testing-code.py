import cv2
from ultralytics import YOLO
import os
import yaml
import torch
import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_boxes(image, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2, label, conf = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_names[label]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def calculate_map50(pred_boxes, gt_boxes):
    thresholds = np.linspace(0.5, 0.95, 10)
    aps = []

    for threshold in thresholds:
        tp, fp, total_gt = 0, 0, len(gt_boxes)
        matched = set()

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched:
                    continue
                iou = calculate_iou(pred_box[:4], gt_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= threshold:
                tp += 1
                matched.add(best_gt_idx)
            else:
                fp += 1

        fn = total_gt - len(matched)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0

        aps.append(precision * recall)

    map50 = np.mean(aps)
    return map50

def detect_and_evaluate(image_folder, label_folder, class_names, output_folder):
    total_iou = 0
    num_correct = 0
    num_predictions = 0
    num_gt_boxes = 0
    correct_images = 0
    total_map50 = 0

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    total_images = len(image_files)
    print(f"Found {total_images} image(s) in '{image_folder}'.")

    for image_file in image_files:
        try:
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image '{image_file}'. Skipping...")
                continue

            resized_image = cv2.resize(image, (640, 640))
            results = model(resized_image)

            label_file = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))
            if not os.path.exists(label_file):
                print(f"Warning: Ground truth for '{image_file}' not found.")
                continue

            gt_boxes = []
            with open(label_file, 'r') as lf:
                for line in lf:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x_min = int((x_center - width / 2) * resized_image.shape[1])
                    y_min = int((y_center - height / 2) * resized_image.shape[0])
                    x_max = int((x_center + width / 2) * resized_image.shape[1])
                    y_max = int((y_center + height / 2) * resized_image.shape[0])
                    gt_boxes.append([x_min, y_min, x_max, y_max, int(class_id)])

            num_gt_boxes += len(gt_boxes)

            if not results or len(results) == 0 or len(results[0].boxes) == 0:
                print(f"No detections found in image '{image_file}'.")
                continue

            pred_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = int(box.cls[0]) if isinstance(box.cls[0], torch.Tensor) else box.cls[0]
                conf = float(box.conf[0])
                pred_boxes.append([x1, y1, x2, y2, label, conf])

            image_correct = False
            for pred_box in pred_boxes:
                best_iou = 0
                for gt_box in gt_boxes:
                    iou = calculate_iou(pred_box[:4], gt_box[:4])
                    best_iou = max(best_iou, iou)
                total_iou += best_iou
                if best_iou >= 0.5:
                    num_correct += 1
                    image_correct = True
                num_predictions += 1

            total_map50 += calculate_map50(pred_boxes, gt_boxes)

            if image_correct:
                correct_images += 1

            output_image = draw_boxes(image.copy(), pred_boxes, class_names)
            output_image_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_image_path, output_image)

        except Exception as e:
            print(f"Error processing image '{image_file}': {str(e)}")

    precision = num_correct / num_predictions if num_predictions > 0 else 0
    recall = num_correct / num_gt_boxes if num_gt_boxes > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = correct_images / total_images if total_images > 0 else 0
    average_iou = total_iou / num_predictions if num_predictions > 0 else 0
    map50 = total_map50 / total_images if total_images > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average IoU: {average_iou:.4f}")
    print(f"mAP@50: {map50:.4f}")

custom_model_path = r'/content/drive/MyDrive/Machine-Learning/dataset_yolov5/models/yolov5_100.pt'
model = YOLO(custom_model_path)

data_yaml_path = r'/content/drive/MyDrive/Machine-Learning/dataset_yolov5/data.yaml'
with open(data_yaml_path, 'r') as file:
    data_config = yaml.safe_load(file)
class_names = data_config['names']

test_image_folder = r'/content/drive/MyDrive/Machine-Learning/dataset_yolov5/test/images'
test_label_folder = r'/content/drive/MyDrive/Machine-Learning/dataset_yolov5/test/labels'
output_folder = r'/content/drive/MyDrive/Machine-Learning/dataset_yolov5/outputs/output_1'
os.makedirs(output_folder, exist_ok=True)

detect_and_evaluate(test_image_folder, test_label_folder, class_names, output_folder)
