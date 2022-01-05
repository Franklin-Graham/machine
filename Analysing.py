from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2


model = load_model(r"D:\mydata\outputdata\videomodels")
lb = pickle.loads(open(r"D:\mydata\outputdata\videoclassificationbinarizer.pickle","rb").read())
outputvideo = r"D:\mydata\outputdata\demo_output.avi"
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
queue = deque(maxlen=128)

capture_video = cv2.VideoCapture(r"D:\mydata\outputdata\demovideo.mp4")
writer = None
(Width,Height) = (None,None)

while True:
    (taken, frame) = capture_video.read()
    if not taken:
        break
    if Width is None or Height is None:
        (Width, Height) = frame.shape[:2]

    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype('float32')
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    queue.append(preds)
    results = np.array(queue).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    text = "They are Playing : {}".format(label)
    cv2.putText(output, text, (45,60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,0,0), 5)

    if writer is None:
        fource = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("outputvideo", fource, 30, (Width, Height), True)
    writer.write(output)
    cv2.imshow("in progress", output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
print("finalizing...")
writer.release()
capture_video.release()
