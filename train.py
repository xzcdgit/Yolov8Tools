from ultralytics import YOLO

def train_resume(path:str):
    # Create a new YOLO model from scratch
    model = YOLO(path)
    # Load a pretrained YOLO model (recommended for training)

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=r"workers.yaml", epochs=300, device=0, resume=True)

    # Evaluate the model's performance on the validation set
    #results = model.val()

    # Perform object detection on an image using the model
    #results = model("https://ultralytics.com/images/bus.jpg")
    # Export the model to ONNX format
    success = model.export(format="onnx")
    print(success)

def train_restart(path:str):
    # Create a new YOLO model from scratch
    #model = YOLO("yolov8m.yaml")
    model = YOLO(path)
    # Load a pretrained YOLO model (recommended for training)

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=r"boxes-seg.yaml", epochs=10, device=0, pretrained=True)

    # Evaluate the model's performance on the validation set
    #results = model.val()

    # Perform object detection on an image using the model
    #results = model("https://ultralytics.com/images/bus.jpg")
    # Export the model to ONNX format
    success = model.export(format="onnx")
    print(success)
    


if __name__ == "__main__":
    resume_path = 'yolov8s-seg.pt'
    train_restart(resume_path)