import cv2
import imutils
import numpy as np
import os
import warnings
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from load_and_process import preprocess_input
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')

class Emotion_Rec:
    def __init__(self):
      emotion_model_path = 'models/_mini_XCEPTION.113-0.64.hdf5'    
      detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'   
      self.emotion_classifier = load_model(emotion_model_path, compile=False)
      self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]
      self.face_detection = cv2.CascadeClassifier(detection_model_path)
      self.list=[]

    def run(self, frame):
       print('a')
       preds = []
       label = None                                                                    
       (fX, fY, fW, fH) = None, None, None, None                                       
       frameClone = frame.copy()                                                       
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                  
       faces = self.face_detection.detectMultiScale(frame, scaleFactor=1.1,            
                                               minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)   
       if len(faces) > 0:                                                                              
          faces = sorted(faces, reverse=False, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))         
                                                                                                
       for i in range(len(faces)):                                                               
                                                                                                
          (fX, fY, fW, fH) = faces[i]                                                           
                                                                                                
                                                                                                
          roi = gray[fY:fY + fH, fX:fX + fW]                                                    
          roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])                       
          roi = preprocess_input(roi)                                                           
          roi = img_to_array(roi)                                                               
          roi = np.expand_dims(roi, axis=0)                                                     
                                                                                                
                                                                                                
          preds = self.emotion_classifier.predict(roi)[0]                                       
                                                                                                
          label = self.EMOTIONS[preds.argmax()]                                                 
                                                                                                
          cv2.putText(frameClone, label, (fX, fY - 10),                                         
                      cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), 1)                            
          cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)             
          cv2.imwrite("I1.jpg", frameClone)                                                     
          print(preds)       
       
          
       for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):                                     
                                                                                                      
           text = "{}: {:.2f}%".format(emotion, prob * 100)                                                
           self.list.append(text)
                            
#["angry", "disgust", "scared", "happy", "sad", "surprised","neutral"]                                                                                    
           w = int(prob * 300) + 7                                                                         
           #canvas=cv2.rectangle(frameClone, (7, (i * 35) + 5), (w, (i * 35) + 35), (224, 200, 130), -1)   
           canvas=cv2.rectangle(frameClone, (7, (i * 35) + 5), (w, (i * 35) + 35), (224, 200, 130), -1)   
           cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)      
                                                                                                      
                                                                                                      
       frameClone = cv2.resize(frameClone, (600,400))                                                       
       cv2.imwrite("I2.jpg", canvas) 
                                                                               
       return (frameClone)                                                                                     

                                                                   
#frame1 = cv2.imread("happy1.jpeg")
#a=Emotion_Rec()
#emotion_model = Emotion_Rec.run(a,frame1)
#print(emotion_model)






