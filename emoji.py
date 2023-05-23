import tkinter as tk 
from tkinter import * 
import cv2 
from PIL import Image, ImageTk
import os
import numpy as np
import cv2 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import threading 

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

emotion_model.add(MaxPooling2D(pool_size=(2,2)))

emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation = 'relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())

emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))

emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

#Disable use of OpenCL
cv2.ocl.setUseOpenCL(False)

#Create dictionary of emotions 
emotion_dict = {
    0: "   Angry   ", 
    1: "   Disgusted   ", 
    2: "   Fearful   ", 
    3: "   Happy   ", 
    4: "   Neutral   ", 
    5: "   Sad   ", 
    6: "   Surprised   "}

#Generate path
cur_path = os.path.dirname(os.path.abspath(__file__))

#Navigate from current path into emojis folder and pick corresponding emotion
emoji_dist = {
    0: cur_path+"/emojis/angry.png",
    1: cur_path+"/emojis/disgusted.png",
    2: cur_path+"/emojis/fearful.png",
    3: cur_path+"/emojis/happy.png",
    4: cur_path+"/emojis/neutral.png",
    5: cur_path+"/emojis/sad.png",
    6: cur_path+"/emojis/surprised.png",
}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text = [0]
global frame_number

def show_subject(): 
    #Open webcam
    cap1 = cv2.VideoCapture(-1)
    print(cap1)
    #cap1 = cv2.VideoCapture('test_videos/Alexa_White.mp4')
        
    if not cap1.isOpened():
        print("Can't find the camera")
    #Do not create local variable 
    global frame_number
    #Max number of frames so know when to exit
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1 
    if frame_number >= length:
        exit()
    cap1.set(1, frame_number)

    #Flag if read something, frame is actual frame
    flag1, frame1 = cap1.read()
    #Resize 
    frame1 = cv2.resize(frame1, (600,500))
    #Represent box around face of person 
    bounding_box = cv2.CascadeClassifier('/home/emma/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #Adjust scaleFactor and minNeighbors for prediction accuracy 
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor = 1.1, minNeighbors=7)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    if flag1 is None:
        print("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_avatar():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text = emotion_dict[show_text[0]], font = ('arial', 45, 'bold'))

    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(15, show_avatar)

if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()

    #Create labels to contain images/video
    lmain = tk.Label(master = root, padx = 50, bd = 10)
    lmain2 = tk.Label(master = root, bd = 25)
    #Quitting button
    lmain3 = tk.Label(master=root, bd = 20, fg = "#CDCDCD", bg = 'blue', font=("Arial", 30))
    
    #Packing and placing in location 
    lmain.pack(side=LEFT)
    lmain.place(x = 50, y = 250)
    lmain3.pack()
    lmain3.place(x = 1025, y = 240)
    lmain2.pack(side=RIGHT)
    lmain2.place(x = 800, y = 250)

    root.title("Translating Realtime Human Facial Expressions to an Emoji using a Trained CNN")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitButton = Button(root, text = 'Quit', fg = "red", command = root.destroy, font = ('arial', 30, 'bold')).pack(side = BOTTOM)

    #Threading allows for parallel computing 
    threading.Thread(target = show_subject).start()
    threading.Thread(target = show_avatar).start()
    root.mainloop()
