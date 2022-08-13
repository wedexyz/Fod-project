

import numpy as np
import cv2
#from sys import modules
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='siap\\best (1).pt') # or yolov5m, yolov5l, yolov5x, custom
#colors = np.random.uniform(0, 255, size=(len(1), 3))

cap = cv2.VideoCapture('video\\sampel_Trim.mp4')





while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame,(400,400))
    #print(frame.shape)
    detections = model(frame[..., ::-1])
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    #print(results)

    for result in results:
        
                con = result['confidence']
                cs  = result['name']
                x1  = int(result['xmin'])
                y1  = int(result['ymin'])
                x2  = int(result['xmax'])
                y2  = int(result['ymax'])

            
                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 0, 0),2)
                cv2.rectangle(frame, (x1-10, y1), (x2+20, y2-25),(0,0,0), cv2.FILLED)
                cv2.putText(frame, str(cs) , (x1-10, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255),1)
                cv2.putText(frame, str(float(np.around(con, 1))) , (x1+10, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
               

    cv2.imshow('frame',frame)
    #results.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

