import cv2
import numpy as np
import yaml

output = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

with open(r'save.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)

print(parameters)
cv2.namedWindow("Trak")

def callback(x):
    pass

video = cv2.VideoCapture("20211128_143232.mp4")

cv2.createTrackbar("Hue Low", "Trak", parameters["min"]["H"], 180, callback)
cv2.createTrackbar("Sat Low", "Trak", parameters["min"]["S"], 255, callback)
cv2.createTrackbar("Val Low", "Trak", parameters["min"]["V"], 255, callback)

cv2.createTrackbar("Hue Hi", "Trak", parameters["max"]["H"], 180, callback)
cv2.createTrackbar("Sat Hi", "Trak", parameters["max"]["S"], 255, callback)
cv2.createTrackbar("Val Hi", "Trak", parameters["max"]["V"], 255, callback)

while True:
    ret, frame = video.read()
    if ret == False:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame2 = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos("Hue Low", "Trak")
    sl = cv2.getTrackbarPos("Sat Low", "Trak")
    vl = cv2.getTrackbarPos("Val Low", "Trak")

    hh = cv2.getTrackbarPos("Hue Hi", "Trak")
    sh = cv2.getTrackbarPos("Sat Hi", "Trak")
    vh = cv2.getTrackbarPos("Val Hi", "Trak")


    lb = np.array([hl, sl, vl])
    hb = np.array([hh, sh, vh])

    mask = cv2.inRange(frame, lb, hb)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel,iterations = 2)
    mask = cv2.dilate(mask, kernel,iterations = 4)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    pantolon = []
    biggest_area = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > biggest_area:
            biggest_area = cv2.contourArea(cnt)
            pantolon = cnt

    mask = np.zeros(frame.shape, np.uint8)

    cv2.drawContours(mask, [pantolon], -1, (255, 255, 255), -1)

    mask = cv2.blur(mask, (18, 18))
    mask_inv = cv2.bitwise_not(mask)

    colored_img = (frame2 / (255.0)) * (mask / (255.0))
    colored_img = np.array(colored_img * 255, np.uint8)

    gray = cv2.cvtColor(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    gray = (gray / (255.0)) * (mask_inv / (255.0))
    gray = np.array(gray * 255, np.uint8)

    result = cv2.addWeighted(colored_img, 1, gray, 0.7, 0)

    cv2.imshow("frame", result)
    output.write(result)
    if ord("q") == cv2.waitKey(1):
        break


output.release()
parameters["min"]["H"] = hl
parameters["min"]["S"] = sl
parameters["min"]["V"] = vl

parameters["max"]["H"] = hh
parameters["max"]["S"] = sh
parameters["max"]["V"] = vh

with open(r'save.yaml', 'w') as file:
    documents = yaml.dump(parameters, file)