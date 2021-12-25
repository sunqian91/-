import tkinter as tk 
import tkinter.filedialog
import cv2
from main import *
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

def choose_file():
    selectFileName = tk.filedialog.askopenfilename(title='选择文件')  
    e.set(selectFileName)
    
def show(e_entry): 
    img = cv2.imread(e_entry.get())
    print(e_entry.get())
    a=Emotion_Rec()
    frameClone = Emotion_Rec.run(a,img)
    
    cv2.imshow("PICTURE",frameClone)
    cv2.waitKey(0)


    
def window(): 
    root = tk.Tk()
    root.geometry('650x450+150+100')
    root.title('test')
    root.resizable(False, False)
 
    global e
    e = tk.StringVar() 
    e_entry = tk.Entry(root, width=68,textvariable=e)
    #e_entry.pack()

    sumbit_btn = tk.Button(root,text="选择文件",bg='yellow',command = choose_file)
    sumbit_btn.pack()
    


    show_btn = tk.Button(root,text= '查看结果',bg = 'blue',\
                             command = lambda :show(e_entry))
    show_btn.pack()
  
    root.mainloop() 
        
 
window()