import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "yolov8n_float32.tflite"
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Check if model files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in the current directory.")
if not os.path.exists(HAAR_CASCADE_PATH):
    raise FileNotFoundError(f"Haar Cascade file '{HAAR_CASCADE_PATH}' not found.")

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

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

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Prepare frame for YOLO detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, tuple(input_shape))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run YOLO inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get YOLO results
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Process YOLO detections
    h, w, _ = frame.shape  # Get original frame size
    detected_any = False
    
    for det in output[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        if conf < 0.5:  # Increased confidence threshold
            continue
            
        detected_any = True
        # Clip coordinates to [0, 1] before scaling
        x1 = int(np.clip(x1, 0, 1) * w)
        y1 = int(np.clip(y1, 0, 1) * h)
        x2 = int(np.clip(x2, 0, 1) * w)
        y2 = int(np.clip(y2, 0, 1) * h)
        
        class_id = int(cls)
        label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else str(class_id)
        
        # Draw detection only if it's a person or if confidence is high
        if label == "person" or conf > 0.7:
            color = (0, 255, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display output
    cv2.imshow('Combined Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
