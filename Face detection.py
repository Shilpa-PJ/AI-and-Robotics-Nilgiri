#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[3]:


# Start video capture from default camera
cap = cv2.VideoCapture(0)


# In[6]:


while True:
    # Read frame from video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding box around detected faces and track them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display frame with face detection and tracking
    cv2.imshow('Face Detection and Tracking', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[5]:





# In[ ]:




