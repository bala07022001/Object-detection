import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# Load pretrained DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Object detection function with visualization
def detect_objects(image):
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process outputs
    target_size = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_size, threshold=0.9)[0]
    
    draw = ImageDraw.Draw(image)
    detected_labels = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_text = model.config.id2label[label.item()]
        detected_labels.append(label_text)
        
        box = [round(i, 2) for i in box.tolist()]
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0), label_text, fill="red")
    
    # Generate label string
    if detected_labels:
        label_summary = ", ".join(set(detected_labels))
        label_output = f"Detected objects: {label_summary}"
    else:
        label_output = "No recognizable objects detected."

    return image, label_output

# Gradio interface with image and text output
gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="Object Detection with Bounding Boxes (DETR)",
    description="Upload an image to detect and visualize objects using DETR."
).launch()
