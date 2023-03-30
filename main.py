import cv2 
import numpy as np
import time

net = cv2.dnn.readNet('./net/yolov3.weights', './net/yolov3.cfg')
classes = []

with open("./labels/coco.txt") as f:
    classes = [x.strip() for x in f.readlines()]
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(low=0, high=255, size=(len(classes),3))

c = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

start = time.time()
f_id = 0

run = True

while run:
    _, frame = c.read()
    f_id += 1
    w, h, ch = frame.shape
    print(frame.shape)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(320,320), mean=(0,0,0), swapRB=True, crop=False)
    
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    if f_id % 3 == 1:
        c_ids=[]
        confidences = []
        boxes=[]
        
        for out in outs:
            for detect in out:

                scores = detect[5:]
                
                c_id = np.argmax(scores)
                print(c_id)
                confidence = scores[c_id]

                if confidence > 0.2:

                    center_x = int(detect[0] * w)
                    center_y = int(detect[1] * h)

                    x = int((center_x - w)//2)
                    y = int((center_y - h)//2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    c_ids.append(c_id)

                    indexes = cv2.dnn.NMSBoxes(boxes,confidences,score_threshold=0.4, nms_threshold=0.8, top_k=1)
    st =''
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[c_ids[i]])
            st += label
            confidence = confidences[i]
            color = colors[c_ids[i]]
            cv2.rectangle(frame,(x ,y),(x + int(w), y + int(h)), color, thickness=2)
            cv2.putText(frame,label + " " + str(round(confidence,2)),(x,y+30),font,2,color,3)
        print(st, boxes[i])

    elapsed_time = time.time() - start
    fps = f_id/elapsed_time

    cv2.putText(frame,"FPS: " + str(round(fps,2)),(40,50),font,3,(0,0,0),3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        c.release()
        cv2.destroyAllWindows()
        run = False
        break

