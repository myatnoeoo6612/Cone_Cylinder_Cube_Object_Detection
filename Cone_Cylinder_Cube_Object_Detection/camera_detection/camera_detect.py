import cv2
from yolo_predictions import YOLO_Pred

yolo = YOLO_Pred('/home/myat/my_pre/best.onnx','/home/myat/my_pre/data.yaml')

cap = cv2.VideoCapture(2)


while True:
    ret, frame = cap.read()
    if ret == False:
        print('unable to read video')
        break
        
    pred_image = yolo.predictions(frame)
    
    cv2.imshow('YOLO',pred_image)
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()
cap.release()