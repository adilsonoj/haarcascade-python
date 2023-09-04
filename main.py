import cv2
import time
from imutils.video import VideoStream
import imutils

vs = VideoStream(src=0).start()
time.sleep(2.0)


classificador = cv2.CascadeClassifier(r'haarcascades/haarcascade_eye.xml')

while True:
  # check, img = camera.read()
  img = vs.read()
  img = imutils.resize(img, width=500)

  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # retorna as coordenadas da imagem treinada
  olhos = classificador.detectMultiScale(imgGray, minSize=(15,15), scaleFactor=1.1, minNeighbors=10, flags=cv2.CASCADE_SCALE_IMAGE)

  # print(olhos)
  for x,y,l,a in olhos:
    # cv2.rectangle(img, (x,y),(x+l,y+a), (255,0,0), 2)
    ptA = (x,y)
    ptB = (x+l,y+a)
    cv2.rectangle(img, ptA, ptB, (0, 0, 255), 2)

  cv2.imshow('Cam1', img)
  # cv2.imshow('Gray', imgGray)

  key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
  if key == ord("q"):
	  break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()