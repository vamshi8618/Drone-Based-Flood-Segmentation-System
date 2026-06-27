import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class YOLODetector:
    def __init__(self,
                 cfg_path='models/yolov3.cfg',
                 weights_path='models/yolov3.weights',
                 names_path='models/coco.names',
                 conf_threshold=0.5,
                 nms_threshold=0.4):
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.names_path = names_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        if not os.path.exists(cfg_path) or not os.path.exists(weights_path) or not os.path.exists(names_path):
            raise FileNotFoundError('YOLO model files not found. Place cfg, weights, and names under models/.')

        self.class_names = self._load_class_names(names_path)
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_layers = self._get_output_layers()

    def _load_class_names(self, names_path):
        with open(names_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        try:
            return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        except AttributeError:
            return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.class_names[class_ids[i]]
                })
        return detections

    def draw_detections(self, frame, detections):
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det['box']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{det['class_name']}:{det['confidence']:.2f}"
            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated


class UNetSegmenter:
    def __init__(self, model_path='unet_model.h5'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'UNet model not found at {model_path}')
        self.model = load_model(model_path)

    def predict_mask(self, frame):
        image = cv2.resize(frame, (256, 256))
        image = image.astype('float32') / 255.0
        batch = np.expand_dims(image, axis=0)
        prediction = self.model.predict(batch)
        mask = prediction[0, ..., 0]
        mask = (mask > 0.5).astype(np.uint8) * 255
        return cv2.resize(mask, (frame.shape[1], frame.shape[0]))


class DroneTracker:
    def __init__(self, target_distance=5.0, yaw_gain=0.8, forward_gain=0.6, throttle_gain=0.4):
        self.target_distance = target_distance
        self.yaw_gain = yaw_gain
        self.forward_gain = forward_gain
        self.throttle_gain = throttle_gain
        self.reference_height = 150

    def estimate_distance(self, bbox_height):
        if bbox_height <= 0:
            return self.target_distance
        return self.target_distance * (self.reference_height / bbox_height)

    def compute_control(self, detection, frame_size):
        if not detection:
            return {'forward': 0.0, 'yaw': 0.0, 'throttle': 0.0, 'roll': 0.0}

        frame_width, frame_height = frame_size
        x, y, w, h = detection['box']
        cx = x + w / 2.0
        cy = y + h / 2.0

        error_x = cx - frame_width / 2.0
        error_y = cy - frame_height / 2.0
        norm_x = clamp(error_x / (frame_width / 2.0), -1.0, 1.0)
        norm_y = clamp(error_y / (frame_height / 2.0), -1.0, 1.0)

        distance = self.estimate_distance(h)
        distance_error = distance - self.target_distance
        forward = clamp(distance_error / self.target_distance * self.forward_gain, -1.0, 1.0)
        yaw = clamp(norm_x * self.yaw_gain, -1.0, 1.0)
        throttle = clamp(-norm_y * self.throttle_gain, -1.0, 1.0)

        return {
            'forward': forward,
            'yaw': yaw,
            'throttle': throttle,
            'roll': 0.0,
            'estimated_distance': float(distance)
        }

    def choose_primary_target(self, detections, class_name='person'):
        candidates = [d for d in detections if d['class_name'] == class_name]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d['box'][2] * d['box'][3])


class FlightPipeline:
    def __init__(self, yolo_detector, segmenter, tracker):
        self.detector = yolo_detector
        self.segmenter = segmenter
        self.tracker = tracker

    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        mask = self.segmenter.predict_mask(frame)
        target = self.tracker.choose_primary_target(detections)
        control = self.tracker.compute_control(target, (frame.shape[1], frame.shape[0]))
        return {
            'detections': detections,
            'mask': mask,
            'target': target,
            'control': control
        }
