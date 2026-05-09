# Handwritten Digit Recognition System using CNN

## Project Description

This project is a handwritten digit recognition system built using a
Convolutional Neural Network (CNN) and trained on the MNIST handwritten
digit dataset.

The system allows a user to upload their own handwritten digit image
through a desktop GUI application and predicts the digit (0--9). The
purpose of this project is to understand how deep learning can be used
for image classification and how it can be integrated into a
user-friendly application.

The project covers the complete machine learning pipeline including
dataset loading, preprocessing, model training, evaluation, prediction,
and deployment through a graphical interface.

------------------------------------------------------------------------

## Objectives

The main objectives of this project are:

-   Train a CNN model using the MNIST dataset
-   Achieve high accuracy in handwritten digit classification
-   Develop a GUI for user interaction
-   Allow users to browse and upload handwritten digit images
-   Preprocess uploaded images properly
-   Predict digits from 0--9
-   Display the prediction result clearly
-   Maintain clean and documented code using GitHub

------------------------------------------------------------------------

## Technologies Used

This project is built using:

-   Python
-   PyTorch
-   Tkinter
-   NumPy
-   Pillow (PIL)
-   Matplotlib
-   MNIST Dataset
-   Git
-   GitHub

------------------------------------------------------------------------

## Dataset Information

The model is trained on the **MNIST Handwritten Digit Dataset**.

Dataset details:

-   Training images: 60,000
-   Testing images: 10,000
-   Image size: 28 × 28 pixels
-   Total classes: 10 (digits 0--9)

The dataset is downloaded automatically while training.

------------------------------------------------------------------------

## CNN Architecture

The CNN model consists of:

-   Convolution Layer (32 filters)
-   Convolution Layer (64 filters)
-   Max Pooling Layer
-   Fully Connected Layer
-   Output Layer (10 classes)

This architecture helps in learning image patterns and classifying
digits accurately.

------------------------------------------------------------------------

## Functional Requirements Covered

This project covers all required functionalities:

### Dataset Handling

-   Loading MNIST dataset
-   Preprocessing dataset

### CNN Model Implementation

-   Convolution layers
-   Pooling layer
-   Dense layers
-   Model training

### Model Evaluation

-   Accuracy calculation
-   Loss calculation

### GUI Features

-   Upload image option
-   Image display
-   Prediction display

### Image Preprocessing

-   Grayscale conversion
-   Resize to 28×28
-   Normalization

### Prediction Logic

-   Digit prediction from uploaded image

### GitHub Requirements

-   Full project uploaded to GitHub
-   Public repository

------------------------------------------------------------------------

## Project Structure

``` text
handwritten-digit-recognition/
│
├── app.py
├── train_model.py
├── mnist_cnn_model.pth
├── requirements.txt
├── README.md
├── screenshots/
│   ├── training_output.png
│   └── gui_result.png
└── .gitignore
```

------------------------------------------------------------------------

## Installation and Setup

### Step 1: Clone the repository

``` bash
git clone https://github.com/AksharaReddyy24-del/handwritten-digit-recognition.git
```

### Step 2: Open project folder

``` bash
cd handwritten-digit-recognition
```

### Step 3: Create virtual environment

``` bash
python -m venv venv
```

### Step 4: Activate virtual environment

For Windows:

``` bash
venv\Scripts\activate
```

### Step 5: Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## How to Run the Project

### Train the Model

Run:

``` bash
python train_model.py
```

This will:

-   Download the MNIST dataset
-   Train the CNN model
-   Evaluate performance
-   Save the model

------------------------------------------------------------------------

### Run the GUI

Run:

``` bash
python app.py
```

This will open the GUI application.

Steps:

1.  Click **Upload Image**
2.  Select your handwritten digit image
3.  View the uploaded image
4.  Get the predicted digit result

------------------------------------------------------------------------

## Model Performance

The model was trained for 5 epochs.

Final Test Accuracy:

**98.73%**

The model performs well on both test data and user-generated handwritten
images.

------------------------------------------------------------------------

## Screenshots

### Training Output

![Training Output](screenshots/training_output.png)

### GUI Result

![GUI Result](screenshots/gui_result.png)

------------------------------------------------------------------------

## Learning Outcomes

Through this project, I learned:

-   How CNN models work
-   How image preprocessing works
-   How to train and evaluate deep learning models
-   How to build GUI applications
-   How to connect ML models with GUI
-   How to manage projects using GitHub

------------------------------------------------------------------------

## Future Improvements

Possible improvements:

-   Better GUI design
-   Drawing pad for live digit input
-   Confidence score display
-   Web-based deployment

------------------------------------------------------------------------

## GitHub Repository

Repository Link:

https://github.com/AksharaReddyy24-del/handwritten-digit-recognition.git

------------------------------------------------------------------------

## Author

Name: Akshara Reddy

Project: Final Project -- Handwritten Digit Recognition System
