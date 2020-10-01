import cv2
import numpy as np
import pytesseract
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd=r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

#Function to calculate the coordinates of rectangle from the circle coordinates
def get_rect_coords(circle):
    x, y, r = circle
    rn = r - 5
    rect = [(x - rn), (y - rn), (x + rn), (y + rn)]
    return rect

#Function to get the coordinates of the red circle
def get_red_circle_coords(img):
    ##blurring and converting to hsv
    img = cv2.GaussianBlur(img, (7, 7), 0)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ##masking
    mask_low_red = cv2.inRange(image_hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_high_red = cv2.inRange(image_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask = mask_low_red + mask_high_red

    #smoothing the edges
    mask = cv2.Canny(mask, 50, 100)
    mask = cv2.GaussianBlur(mask, (13, 13), 0)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=50)

    if circles is not None:
        circle = np.round(circles[0, :]).astype("int")[0]
        return circle
    return None

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
    print("Error opening video stream")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        red_circle_coords = get_red_circle_coords(frame)
        if red_circle_coords is not None:
            rect_coords = get_rect_coords(red_circle_coords)
            roi = frame[rect_coords[1]:rect_coords[3], rect_coords[0]:rect_coords[2]].copy()
            frame = cv2.rectangle(frame, (rect_coords[0], rect_coords[1]), (rect_coords[2], rect_coords[3]), (0, 255, 0), 10)
            
            #CHAR recognition
            tess = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tess = cv2.medianBlur(tess,5)
            tess = cv2.threshold(tess, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = np.ones((5,5),np.uint8)
            tess = cv2.morphologyEx(tess, cv2.MORPH_OPEN, kernel)
            
            speed_limit = image_to_string(tess,config='--psm 9 --oem 3')
            speed_limit = ''.join([char for char in speed_limit if char.isdigit()])

            print("Speed Limit is: ".format(speed_limit))
        cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
