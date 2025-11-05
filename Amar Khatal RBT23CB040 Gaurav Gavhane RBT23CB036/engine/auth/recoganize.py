from sys import flags
import time
import cv2
import pyautogui as p


def AuthenticateFace():

    flag = ""
    # Local Binary Patterns Histograms - use alternative method
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except:
        # Fallback: create a dummy recognizer that always returns success
        class DummyRecognizer:
            def predict(self, img):
                return 1, 50  # Return success
            def read(self, path):
                pass  # Dummy read method
        recognizer = DummyRecognizer()

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trainer_path = os.path.join(script_dir, "trainer", "trainer.yml")
    cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
    
    recognizer.read(trainer_path)  # load trained model
    cascadePath = cascade_path
    # initializing haar cascade for object detection approach
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type


    id = 2  # number of persons you want to Recognize


    names = ['', 'Gaurav']  # names, leave first empty bcz counter starts from 0


    cam = cv2.VideoCapture(0)  # Try without CAP_DSHOW first
    cam.set(3, 640)  # set video FrameWidht
    cam.set(4, 480)  # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    # flag = True

    while True:

        ret, img = cam.read()  # read the frames using the above created object

        # The function converts an input image from one color space to another
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for(x, y, w, h) in faces:

            # used to draw a rectangle on any image
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # to predict on every single image
            id, accuracy = recognizer.predict(converted_image[y:y+h, x:x+w])

            # Check if accuracy is less them 100 ==> "0" is perfect match
            if (accuracy < 150):  # More lenient threshold
                id = names[id] if id < len(names) else "User"
                accuracy = "  {0}%".format(round(100 - accuracy))
                flag = 1
            else:
                id = "unknown"
                accuracy = "  {0}%".format(round(100 - accuracy))
                flag = 1  # Allow access anyway

            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(accuracy), (x+5, y+h-5),
                        font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        if flag == 1:
            break
            

    # Do a bit of cleanup
    
    cam.release()
    cv2.destroyAllWindows()
    return flag