import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
model = model_from_json(open("m2fcc.json", "r").read())

model.load_weights('m2fcc.h5')
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()

for i in range(1,16):   
    
    path=r"c:\users\RISHABH\Desktop\faces\{}.jpg".format(i)
    img = image.load_img(path ,grayscale=True , target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    custom = model.predict(x)
    emotion_analysis(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);
    

    plt.gray()
    plt.imshow(x)
    plt.show()
#---
