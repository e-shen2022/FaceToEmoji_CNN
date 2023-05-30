import tkinter as tk 
from tkinter import * 
import cv2 
from PIL import Image, ImageTk
import os
from cv2 import CAP_V4L2
import numpy as np
import cv2 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import threading 
import time

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
#Initializes array with 0s that will pass in image's RGB values 
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
#Global emoji dictionary index shared between threads
show_text = [0] 
show_text[0] = 4
# Create an event
switch_thread_event = threading.Event()
stop_event = threading.Event()

# Debug counters
subject_count = 0 
avatar_count = 0

def show_subject():
    global subject_count

    while not stop_event.is_set():
        # Wait for the event to be set
        if not switch_thread_event.wait(5):
            print("Subject Timeout occurred!")
            break

        #Open webcam
        cap1 = cv2.VideoCapture(0)

        if not cap1.isOpened():
            print("Can't find the camera")
        else:
            print("Opened Camera")

        # frame 1 captures the video frame by frame, flag1 returns frame status 
        flag1, frame1 = cap1.read()

        # Comment this out, this creates a frame outside my display: Display the resulting frame
        #cv2.imshow('frame', frame1)

        #Resize 
        frame1 = cv2.resize(frame1, (600,500))

        #Represent box around face of person, classifier is used to capture faces in a frame 
        bounding_box = cv2.CascadeClassifier('C:\Emojify\data\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        #Note: Adjust scaleFactor and minNeighbors for prediction accuracy 
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor = 1.1, minNeighbors=7)
        for (x, y, w, h) in num_faces: 
            #Rectangle drawn around each detected face
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            #Cropped face image fed into emotion model for prediction
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            #Retrieve emotion label & display
            cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            show_text[0]=maxindex
            current_time_ms = time.time_ns() // 10**6
            subject_count = subject_count + 1 
            print("Current time for subject", current_time_ms, show_text[0], subject_count)

        print("flag1", flag1)

        if flag1 is None:
            print("Major error!")
        elif flag1 == True:
            global last_frame1
            last_frame1 = frame1.copy()
            pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(pic)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)

        # After loop release webcam to be used in this program
        cap1.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        #Never prints bc webcam not released 
        print('webcam destroyed')

        time.sleep(1)
        root.update()

        # Reset the switch_thread_event
        switch_thread_event.clear()
        switch_thread_event.set()

    print("Subject thread is finished")

def show_avatar():
    global avatar_count

    while not stop_event.is_set():
        # Wait for the switch_thread_event to be set
        if not switch_thread_event.wait(5):
            print("Avatar Timeout occurred!")
            break

        emoji_index = show_text[0]
        avatar_count = avatar_count + 1 
        current_time_ms = time.time_ns() // 10**6
        print("Current time for avatar", current_time_ms, emoji_index, avatar_count)

        frame2 = cv2.imread(emoji_dist[emoji_index])
        pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(pic2)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk2 = imgtk2
        lmain3.configure(text = emotion_dict[emoji_index], font = ('arial', 45, 'bold'))
        lmain2.configure(image=imgtk2)
        
        time.sleep(1)
        root.update()

        # Reset the switch_thread_event
        switch_thread_event.clear()
        switch_thread_event.set()
    
    print ("Avatar thread is finished")

def stop_threads():
    global stop_event
    global switch_thread_event

    stop_event.set()
    switch_thread_event.clear()

def wrapper_quit():
    # Signal the worker thread to quit
    stop_threads()
    print("After stop camera thread ", stop_event.is_set(), switch_thread_event.is_set())
    root.destroy()

if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()
    
    #Create labels to contain images/video
    #Human video
    lmain = tk.Label(master = root, padx = 50, bd = 10)
    #Emoji
    lmain2 = tk.Label(master = root, bd = 25)
    #Quit button for entire program
    lmain3 = tk.Label(master=root, bd = 20, fg = "#CDCDCD", bg = 'blue', font=("Arial", 30))
    
    #Packing and placing in location 
    lmain.pack(side=LEFT)
    lmain.place(x = 30, y = 200)
    lmain3.pack()
    lmain3.place(x = 1000, y = 600)
    lmain2.pack(side=RIGHT)
    lmain2.place(x = 700, y = 200)

    root.title("Translating Realtime Human Facial Expressions to an Emoji using a Trained CNN")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    #How when click quit button to also quit webcam? 
    switch_thread_event.set() 
    subject_thread = threading.Thread(target = show_subject)
    avatar_thread = threading.Thread(target = show_avatar)
    exitButton = Button(root, text = 'Quit', fg = "red", command = wrapper_quit, font = ('arial', 30, 'bold')).pack(side = TOP)

    #Threading allows for parallel computing 
    subject_thread.start()
    avatar_thread.start()

    print("Before main loop")
    root.mainloop()
