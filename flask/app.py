from flask import Flask,render_template,request
from flask.globals import request

import tensorflow as tf
import keras
from keras.models import load_model
import cv2
import numpy as np 
from PIL import Image
import imutils
from werkzeug.datastructures import LanguageAccept



flag=0
app = Flask(__name__)
# model = load_model('model.h5')

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/' +imagefile.filename
    imagefile.save(image_path)


    #load_model
    def mean_squared_loss(x1,x2):
        difference=x1-x2
        a,b,c,d,e=difference.shape
        n_samples=a*b*c*d*e
        sq_difference=difference**2
        Sum=sq_difference.sum()
        distance=np.sqrt(Sum)
        mean_distance=distance/n_samples
        return mean_distance

    model=load_model("model.h5")

    cap = cv2.VideoCapture(image_path)
    print(cap.isOpened())


    while cap.isOpened():
        imagedump=[]
        ret,frame=cap.read()

        for i in range(10):
            ret,frame=cap.read()
            if ret == False:
                cap.release()
                break
            #image = imutils.resize(frame,width=700,height=600)
            image = cv2.resize(frame,(700,600),cv2.INTER_AREA)

            frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
            gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray=(gray-gray.mean())/gray.std()
            gray=np.clip(gray,0,1)
            imagedump.append(gray)
            image = imutils.resize(frame,width=700,height=600)

        imagedump=np.array(imagedump)

        imagedump.resize(227,227,10)
        imagedump=np.expand_dims(imagedump,axis=0)
        imagedump=np.expand_dims(imagedump,axis=4)

        output=model.predict(imagedump)

        loss=mean_squared_loss(imagedump,output)

        # if frame.any()==None:
        #     print("none")

        if cv2.waitKey(12) & 0xFF==ord('q'):
            break
        if loss>0.00068:
            flag=1
            print('Abnormal Event Detected')
            cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
        else:
            flag=0
            print('Normal Event Detected')
            cv2.putText(image,"Normal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

        cv2.imshow("video",image)

    cap.release()
    cv2.destroyAllWindows()

        

        







    # image = load_img(image_path, target_size=(224,224))
    # image = img_to_array(image)
    # image = image.reshape((1,image.shape[0], image.shape[1],image.shape[2]))
    # image = preprocess_input(image)
    # yhat = model.predict(image)
    # label = decode_predictions(yhat)
    # label = label[0][0]

    



    return render_template('index.html',data=flag)



if __name__ == "__main__":
    app.run(port=3000,debug=True)
