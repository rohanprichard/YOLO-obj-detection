import cv2 
import numpy as np
import time
from cv2.dnn import *

net = cv2.dnn.readNet('./net/yolov3-tiny.weights', './net/yolov3-tiny.cfg')
classes = []

with open("./labels/coco.txt") as f:
    classes = [x.strip() for x in f.readlines()]
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes),3))

c = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

start = time.time()
f_id = 0

run = True

while run:
    _, frame = c.read()
    f_id += 1
    he, wi, ch = frame.shape

    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320,320), mean=(0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    c_ids=[]
    confidences = []
    boxes=[]
    
    for out in outs:
        for detect in out:
            scores = detect[5:]
            c_id = np.argmax(scores)
            confidence = scores[c_id]

            if confidence > 0.2:
                center_x = int(detect[0] * wi)
                center_y = int(detect[1] * he)

                x = int(center_x - wi//2)
                y = int(center_y - he//2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                c_ids.append(c_id)

                indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[c_ids[i]])
            confidence = confidences[i]
            color = colors[c_ids[i]]

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,thickness=2)
            cv2.putText(frame,label + " " + str(round(confidence,2)),(x,y+30),font,3,color,3)
        
    elapsed_time = time.time() - start
    fps = f_id/elapsed_time

    cv2.putText(frame,"FPS: "+str(round(fps,2)),(10,50),font,4,(0,0,0),3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        c.release()
        cv2.destroyAllWindows()
        run = False
        break
