1. Introduction
Face recognition technology is transforming attendance management by automating the process of identifying and recording individuals. Google Teachable Machine simplifies the creation of machine learning models for face recognition without requiring advanced programming skills. This report outlines the use of Google Teachable Machine to build a face recognition-based attendance system, including a demonstration with an attendance list.

2. Overview of Google Teachable Machine
Google Teachable Machine is a no-code, web-based platform that allows users to create and train machine learning models for image, sound, and pose recognition. It is particularly suitable for beginners and small-scale projects.

Features:
User-friendly Interface: Simplifies the training process.
Real-time Feedback: Provides immediate accuracy metrics.
Export Options: Supports TensorFlow.js, TensorFlow Lite, and other formats.

3. Building a Face Recognition Attendance System
Data Collection
Collect facial images of individuals in various poses and lighting conditions.
Organize the images into labeled classes representing each person (e.g., John, Mary, Alice).

Model Training
Import the collected images into Google Teachable Machine.
Assign labels to each class (e.g., Employee Names).
Train the model using default or custom settings (e.g., epochs, batch size).

Testing the Model
Test the model by uploading unseen images of individuals.
Evaluate the accuracy and adjust the dataset if necessary.

Export and Deployment
Export the trained model in TensorFlow.js format.
Integrate the model into a Python-based application for attendance tracking.

4. Attendance System Workflow
Face Detection: Use the model to identify individuals in real-time or from uploaded images.
Attendance Recording: Once an individual is recognized, their name and timestamp are recorded in an attendance database.
Display Attendance List: Generate a table or CSV file showing the attendance for a specific date.
