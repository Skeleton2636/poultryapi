from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import os, shutil
import cv2
import numpy as np

# Paths of yolo files api\model_files\yolov4-tiny-custom_best.weights
weightsPath = "model_files\yolov4-tiny-custom_best.weights"
configPath = "model_files\yolov4-tiny-custom.cfg"
labelsPath = "model_files\obj.names"

# Load Yolo
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#save all the names in file o the list classes
classes = []
with open(labelsPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]

LABELS = open(labelsPath).read().strip().split("\n")


def yolo(image):

    #get layers of the network
    layer_names = net.getLayerNames()

    #Determine the output layer names from the YOLO model 
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Read image using cv2
    img = cv2.imread(image)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    
    #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)    
    labels = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            #(x, y) = (boxes[i][0], boxes[i][1])
            #(w, h) = (boxes[i][2], boxes[i][3])
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            text = "{}: {:.4f}".format(
                    LABELS[class_ids[i]], confidences[i])
            labels.append(text)
        return labels


    # if len(confidences)>0 and len(class_ids)>0:
    #     index = confidences.index(max(confidences))
    #     ls = [confidences[index], class_ids[index]]
    #     return ls
    #     # return {"confidence": confidences[index],
    #     #         "class": class_ids[index]}   
    # else:
    #     return [0,0]
    #     # return {"confidence": 0,
    #     #         "class": 0}

app = FastAPI()


UPLOAD_FOLDER = os.getcwd() + '/tmp/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.get("/")
def root():
    return {"health":200}


@app.post("/v1/image")
async def total_api_calls(request: Request, image: UploadFile = File(...)):
    if request.method == 'POST':
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        os.chdir(UPLOAD_FOLDER)
        with open("temp." + str(image.filename.split(".")[-1]), "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        file = [file for file in os.listdir(os.getcwd())][0]
        result = yolo(file)
        return {"result":result}