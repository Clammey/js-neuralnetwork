#import packages

import cv2

face_classifier = cv2.CascadeClassifier( # Face Detection model
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
#define video_capture
video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:
          # Capture webcam frame
    _, img = video_capture.read() 
    img = cv2.flip(img, 1) 
          
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # 

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame
        
    cv2.imshow(
        "Face Detection for PMC", video_frame
    )  # displays window named "Face Detection for PMC" of webcam

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
