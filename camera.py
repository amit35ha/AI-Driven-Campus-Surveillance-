import os
import cv2
import numpy as np
import tensorflow as tf

# Suggestion: Rename this file from 'import cv2.py' to something else, e.g., 'yolo_webcam.py' to avoid import conflicts with cv2

MODEL_PATH = "yolov8n_float32.tflite"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in the current directory.")

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input size
input_shape = input_details[0]['shape'][1:3]  # e.g., [640, 640]

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Please check your camera device.")

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Resize and normalize image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, tuple(input_shape))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get results
    output = interpreter.get_tensor(output_details[0]['index'])  # Shape: (1, num_detections, 6) -- may vary by model
    # If you get shape errors, print(output.shape) to debug
    print("Output shape:", output.shape)
    print("First detection:", output[0][0])
    print("All detections in this frame:")
    print(output[0])

    # Post-processing
    h, w, _ = frame.shape  # Get original frame size
    detected_any = False
    for det in output[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        if conf < 0.05:
            continue
        detected_any = True
        # Clip coordinates to [0, 1] before scaling
        x1 = int(np.clip(x1, 0, 1) * w)
        y1 = int(np.clip(y1, 0, 1) * h)
        x2 = int(np.clip(x2, 0, 1) * w)
        y2 = int(np.clip(y2, 0, 1) * h)
        print(f"Detection: class={cls}, conf={conf}, box=({x1},{y1},{x2},{y2})")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = COCO_LABELS[int(cls)] if int(cls) < len(COCO_LABELS) else str(int(cls))
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if not detected_any:
        print("No detections in this frame.")

    # Display output
    cv2.imshow('YOLOv8-TFLite Real-Time Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 