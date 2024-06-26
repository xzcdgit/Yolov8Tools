from ultralytics import YOLO
if __name__ == "__main__":
<<<<<<< HEAD
    # Create a new YOLO model from scratch
    model = YOLO("yolov8s.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8s.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=r"dataset\workers.yaml", epochs=350, device=0)

    # Evaluate the model's performance on the validation set
    #results = model.val()

    # Perform object detection on an image using the model
    #results = model("https://ultralytics.com/images/bus.jpg")
    # Export the model to ONNX format
    success = model.export(format="onnx")
    print(success)
=======
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
>>>>>>> b4a804dc907180b56e261c2f762d7b9e08735c78
