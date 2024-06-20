from ultralytics import YOLO
if __name__ == "__main__":
    # Create a new YOLO model from scratch
    model = YOLO("yolov8n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=r"dataset\boards.yaml", epochs=300, device=0)

    # Evaluate the model's performance on the validation set
    #results = model.val()

    # Perform object detection on an image using the model
    #results = model("https://ultralytics.com/images/bus.jpg")
    # Export the model to ONNX format
    success = model.export(format="onnx")
    print(success)