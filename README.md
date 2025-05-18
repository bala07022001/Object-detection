# Object-detection

This project implements an object detection pipeline using Facebook AI's DETR (DEtection TRansformer) model. It uses Hugging Face Transformers, PyTorch, and Gradio to create a simple web interface where users can upload images and view the detected objects with bounding boxes drawn around them.

Uses facebook/detr-resnet-50 pretrained object detection model.

1. Draws bounding boxes and object labels on uploaded images.

2. Displays a text summary of detected objects.

3. Runs in a clean and interactive Gradio web app.

4. High detection confidence threshold (0.9) ensures only precise results.

REQUIREMENTS:
pip install torch torchvision transformers pillow gradio

How it works:

Image Upload: User uploads an image via the Gradio interface.

Image Processing: The image is processed using Hugging Face's DetrImageProcessor.

Model Inference: The processed image is fed to the DETR model to detect objects.

Post-Processing:

Extracts bounding boxes and class labels.

Draws boxes on the image using PIL.ImageDraw.

Generates a summary sentence of all detected object classes.

Output: Returns:

Annotated image (with boxes and labels)

Text summary like: "Detected objects: person, dog, bicycle"

