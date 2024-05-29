#1 Project Goals/Description

The goal of this project is to create a model capable of detecting human emotion through a realtime web cam and match the expression with a corresponding emoji.

For that we use a dataset containing more than 28700 images that is already classified in one of these 7 categories: angry, disgust, fear, happy, neutral, sad, and surprise.

We are going to create a machine learning algorithm, specifically a Convolutional Neural Network (CNN), with the platform Tensorflow to train the model based on this data to recognize facial expressions and map those same emotions on an emoji.

<img width="868" alt="image" src="https://github.com/e-shen2022/FaceToEmoji_CNN/assets/104966589/4eb0b235-50ab-45a8-a304-d43b34b99fc4">

#2 Tool/Environment Setup

Some tools/topics covered

Language: Python

Deep Neural Networks (Tensorflow)

Python Packages (Keras)

1. VSCode Environment

(a) Create a new folder in File Explorer and name it Project Name in your C Drive (Directly in your OS folder)

(b) Open the folder in VSCODE

(c) Create a new folder called "src" and two new files called "train.py" and "emoji.py".

(d) Now create 2 subfolders under "src" called "data" and "emojis".

(e) Navigate to this dataset on Kaggle and download it. We will be using this dataset to train our model so look around and familiarize yourself with what this data is!

(f) Download and extract the data into the "data" folder. You should now be able to see two subset folders labeled "train" and "test" folders with many pictures under the "data" folder.

We will be filling in the emojis folder later. This is all you need to set up for now!

2. Modules to Install

OpenCV: Otherwie known as Open Source Computer Vision. A library that provides a set of tools/functions to process/analyze images and videos

Numpy: Python library that allows us to use multi-dimensional rrays to store large datasets and use optimized mathematical functions for data analysis

Tensorflow: A very useful tool for machine learning. Takes data, builds a model, trains it, and then lets us use the trained model to make predictions!

Keras: A high-level neural networks API integrated into Tensorflow


