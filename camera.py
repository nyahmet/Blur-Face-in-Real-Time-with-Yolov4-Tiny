#importing necessary libraries
import cv2 as cv
import time
import os

#reading label names from related txt file
class_name = []
with open(os.path.join("project_files",'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

#importing model .weights and config file 
#defining model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny_best.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#defining video source
cap = cv.VideoCapture(0)

#getting video source frame width and height
width  = cap.get(3)
height = cap.get(4)

#defining result video writer parameters
result = cv.VideoWriter('result.mp4', 
                         cv.VideoWriter_fourcc(*'MP4V'),
                         20,(int(width),int(height)))

#defining some parameters will affect detection process
Conf_threshold = 0.2 #related with the detection is acceptable or not
NMS_threshold = 0.6 #related with bounding box size

#initial values of some parameters
frame_counter = 0
starting_time = time.time()

#detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break
    #analysis the video stream based on the model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        if(classid==0): 
            label = class_name[0]
            x, y, w, h = box
            #drawing bounding box around the detected object and blurring it
            frame[y:y+h, x:x+w] = cv.blur(frame[y:y+h, x:x+w] ,(50,50))
            #cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)
            cv.putText(frame, "%" + str(round(scores[0]*100,2)) + " " + label, (box[0], box[1]-10),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        
        
    #writing FPS on screen
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    #cv.putText(frame, f'FPS: {fps}', (20, 50),
               #cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    

    #showing and saving result
    cv.imshow('frame', frame)
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    
#end
cap.release()
result.release()
cv.destroyAllWindows()
