import numpy as np
import os
import torch

def createArray(detectionList):
    arr = []
    for line in detectionList:
        if line != []:
            words = line.split()
            if words[0] == "2" or words[0] == "7":
                #if car or truck add to array
                arr.append([float(words[1]), float(words[2])])
    arr = np.asarray(arr)
    return arr

def load_model():
    modelPath = os.path.join(os.getcwd(), 'yolov5s.pt')
    if os.path.exists(modelPath):
        return torch.hub.load('ultralytics/yolov5', 'custom', path=modelPath, device=0, _verbose=False)
    else:
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', device=0, _verbose=False)

def detect(img, model):
    detections = []
    width = img.width
    height = img.height
    model.classes = [2, 7]
    results = model(img)
    results_df = results.pandas().xyxy[0]
    for i in results_df.index:
        center_x = ((results_df['xmax'][i] + results_df['xmin'][i]) / 2) / width
        center_y = ((results_df['ymax'][i] + results_df['ymin'][i]) / 2) / height
        detection_class = results_df['class'][i]
        detection = [str(detection_class) + " " + str(center_x) + " " + str(center_y)]
        detections = detections + detection
    return detections