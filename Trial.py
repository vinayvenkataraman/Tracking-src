import numpy as np
import cv2

cap = cv2.VideoCapture('Input.mp4')

frame_no = 1030
cap.set(1, frame_no);

isFirstFrame = True
min_area = 500

while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = frame[300:, :]

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if isFirstFrame:
        isFirstFrame = False
        refImage = gray
        continue

    diffImage = cv2.absdiff(refImage, gray)
    thresh = cv2.threshold(diffImage, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=4)

    #(cnts, _, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxarea = 0;
    contourindex = -1;
    for i in range (0, len(contours)):
        if (cv2.contourArea(contours[i]) > maxarea) :
            maxarea = cv2.contourArea(contours[i])
            contourindex = i

    if len(contours) :
        cnt = contours[i]
        #area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    refImage = gray

    #cv2.imshow('Binary Difference Image', thresh)
    cv2.imshow('Player Tracked', frame)

cap.release()
cv2.destroyAllWindows()
