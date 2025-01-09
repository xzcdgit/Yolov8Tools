from ultralytics import YOLO
def train_resume(path:str, data):
    # Create a new YOLO model from scratch
    model = YOLO(path)
    # Load a pretrained YOLO model (recommended for training)

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data, epochs=300, device=0, resume=True)

    # Evaluate the model's performance on the validation set
    #results = model.val()

    # Perform object detection on an image using the model
    #results = model("https://ultralytics.com/images/bus.jpg")
    # Export the model to ONNX format
    success = model.export(format="onnx")
    print(success)

def train_restart(path:str,data):
    # Create a new YOLO model from scratch
    #model = YOLO("yolov8m.yaml")
    model = YOLO(path)
    # Load a pretrained YOLO model (recommended for training)

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=r"C:\Users\24225\AppData\Local\Programs\Python\Python311\Lib\site-packages\ultralytics\cfg\datasets\men_heng_liang_liu_shui_xian.yaml", epochs=400, device=0, pretrained=True)

    # Evaluate the model's performance on the validation set
    #results = model.val()

    # Perform object detection on an image using the model
    #results = model("https://ultralytics.com/images/bus.jpg")
    # Export the model to ONNX format
    success = model.export(format="onnx")
    print(success)
    
if __name__ == "__main__":
    resume_path = 'C:\Code\Python\Yolov8Tools\yolov8s.pt'
    data = r"C:\Users\24225\AppData\Local\Programs\Python\Python311\Lib\site-packages\ultralytics\cfg\datasets\men_heng_liang_liu_shui_xian.yaml"
    train_restart(resume_path, data)