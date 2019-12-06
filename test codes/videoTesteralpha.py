
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt 
def plotters(colle):
    neutral=[]
    angry=[]
    disg=[]
    sad=[]
    happy=[]
    fear=[]
    surp=[]
    xaxis=[]
    for x in range(0,len(colle)):
        neutral.append(colle[x]['neutral'])
        angry.append(colle[x]['angry'])
        disg.append(colle[x]['disgust'])
        sad.append(colle[x]['sad'])
        happy.append(colle[x]['happy'])
        fear.append(colle[x]['fear'])
        surp.append(colle[x]['surprise'])
        xaxis.append(x)
    plt.scatter(xaxis,angry)
    #plt.plot(xaxis,forangy)
    plt.xlabel("time axis")
    plt.ylabel('angry percentage')
    plt.show()
    plt.scatter(xaxis,happy)
    plt.xlabel("time axis")
    plt.ylabel('happy percentage')
    plt.show()
    plt.scatter(xaxis,sad)
    plt.xlabel("time axis")
    plt.ylabel('sad percentage')
    plt.show()
    plt.scatter(xaxis,fear)
    plt.xlabel("time axis")
    plt.ylabel('fear percentage')
    plt.show()
    plt.scatter(xaxis,neutral)
    plt.xlabel("time axis")
    plt.ylabel('neutral percentage')
    plt.show()
    
    
    
    plt.plot(xaxis,angry,color="green")
    #plt.plot(xaxis,forangy)
    plt.xlabel("time axis")
    plt.ylabel('angry percentage')
    plt.show()
    plt.plot(xaxis,happy,color="green")
    plt.xlabel("time axis")
    plt.ylabel('happy percentage')
    plt.show()
    plt.plot(xaxis,sad,color="green")
    plt.xlabel("time axis")
    plt.ylabel('sad percentage')
    plt.show()
    plt.plot(xaxis,fear,color="green")
    plt.xlabel("time axis")
    plt.ylabel('fear percentage')
    plt.show()
    plt.plot(xaxis,neutral,color="green")
    plt.xlabel("time axis")
    plt.ylabel('neutral percentage')
    plt.show()
    
#load model
model = model_from_json(open("xyz.json", "r").read())
#load weights
model.load_weights('xyz.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)
colle=[]
while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
       
      #   print(predictions[0])
        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        mapped=zip(emotions,predictions[0])
        mapped=dict(mapped)
        colle.append(mapped)   
        predicted_emotion = emotions[max_index]
        print(mapped)
       
        ans=emotions[max_index]
        cv2.putText(test_img,ans, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
plotters(colle)

        
