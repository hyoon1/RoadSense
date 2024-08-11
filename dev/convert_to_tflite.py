from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Convert YOLO model to TFLite and run a test.")
parser.add_argument('--model_path', type=str, default='./best.pt', help="Path to the YOLO model (.pt file). Defaults to './best.pt'.")

args = parser.parse_args()

# Extract the model path from arguments
model_path = args.model_path

# Load the best trained model
model = YOLO(model_path)

# Export the model to TFLite format (default is float32)
model.export(format="tflite")

# Automatically find the generated TFLite file within the saved_model directory
model_dir = os.path.dirname(model_path)
tflite_dir = os.path.join(model_dir, 'best_saved_model')
tflite_model_path = os.path.join(tflite_dir, 'best_float16.tflite')

# Check if the TFLite model was saved correctly
if os.path.exists(tflite_model_path):
    print(f"TFLite model saved at {tflite_model_path}")
else:
    print("Failed to save TFLite model or could not locate the file.")

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the interpreter
interpreter.invoke()

# Print confirmation message
print("Model successfully converted to TFLite and test run completed.")
