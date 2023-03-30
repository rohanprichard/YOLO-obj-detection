import cv2
import numpy as np

# Load the SSD model
net = cv2.dnn.readNetFromCaffe('ssd.prototxt', 'ssd.caffemodel')

# Define the classes that the model can detect
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Set the minimum confidence level for a detection
confidence_threshold = 0.5

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to a blob for input to the SSD model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)

    # Set the blob as input to the SSD model
    net.setInput(blob)

    # Perform object detection using the SSD model
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence level of the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            # Extract the class label of the detection
            class_index = int(detections[0, 0, i, 1])

            # Get the bounding box of the detection
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1],
                                                       frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and class label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            label = "{}: {:.2f}%".format(classes[class_index],
                                          confidence * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the detections
    cv2.imshow("Object Detection", frame)

    # Quit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
