####################### Import Packages #######################

import numpy as np 
from tensorflow import keras
#from tensorflow import Adam
from keras.models import Sequential, load_model 
#Dense - Fundamental building block to NN 
#Dropout - Regularizes NN to prevent overfitting 
#Flatten - Convert multidimensional input data (images) into 1D array
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

####################### Train Data ############################

#Define the directories where training/testing data located 
train_dir = 'data/train'
value_dir = 'data/test'

#Divides image pixel values by 255 to scale down pixel values to normalized range between 0 and 1 for NN training
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

#Loads images from train_dir
train_generator = train_datagen.flow_from_directory(
    train_dir,

    #Resize images to 48 x 48 pixels 
    target_size = (48, 48),

    #Number of images processed in each batch 
    batch_size = 64,

    #Convert images to grayscale (reduce dimensionality of input data from RGB to intensity)
    color_mode = "grayscale",

    #Labels for images are categorical values (Ex. Happy, Sad, Surprised, etc)
    class_mode = 'categorical'
    )

validation_generator = validation_datagen.flow_from_directory(
    train_dir,
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
    )

####################### Creating CNN ###########################

#Using the sequential model that builds the CNN layer by layer
emotion_model = Sequential()

#Adding 2 convolutional layers that are responsible for detecting local patterns in the input data
#1st layer has 32 filters of size 3x3 pixels and applies the ReLU activation function, 2nd is the same except has 64 filters that allows the model to extract more complex patterns
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

#Pooling layers: Downsample data to look at small areas of the image (reduces spatial dimensions while retaining important features)
emotion_model.add(MaxPooling2D(pool_size=(2,2)))

#Regularization: Randomly sets input units to 0 to prevent overfitting (learn noise rather than actual signal)
emotion_model.add(Dropout(0.25))

#More convolutional/pooling layers and regularization
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation = 'relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

#Reshapes ouput from previous layers into 1D vector to prepare for the fully connected layers
emotion_model.add(Flatten())

#1st dense layer: 1024 neurons fully connected layer
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))

#2nd dense layer: 7 neurons which is # of possible output classes (emotions)
#Uses softmax activation function to convert final layer's raw predicted values into a probability distribution over the different classes for classification
emotion_model.add(Dense(7, activation='softmax'))

###################### Compile Model ############################

#Prepare model for training by defining how it will measure loss, update its weights, and evaluate its prediction performance
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

#Train the model using a generator
emotion_model_info = emotion_model.fit_generator(
    train_generator,

    #Number of batches processed in each epoch
    steps_per_epoch =28709 // 64,

    #Number of times the entire dataset is passed through the model for training
    epochs=50,

    validation_data=validation_generator,

    #Number of batches to be processed for validation in each epoch
    validation_steps=7178 // 64 
)

#Save learned parameters
emotion_model.save_weights('model.h5')


