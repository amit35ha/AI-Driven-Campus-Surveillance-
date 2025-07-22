from ultralytics import YOLO

# Download YOLOv8n PyTorch model and export to TFLite
model = YOLO('yolov8n.pt')
model.export(format='tflite')

print('Export complete. The file yolov8n_float32.tflite should now be in your directory.') 