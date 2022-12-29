import cv2

img_file = 'car.jpg'
video = cv2.VideoCapture("Street - 136605.mp4")

car_tracker_file = ('cars.xml')
pedestrian_tracker_file = ('haarcascade_fullbody.xml')

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True :

    (read_successful, frame) = video.read()

    if read_successful :
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else :
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrains = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    for (x, y, w, h) in cars :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), )
    
    for (x, y, w, h) in pedestrains :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,225,255), 2)

    cv2.imshow('car detector', frame)

    cv2.waitKey(1)


print("Code Completed")
