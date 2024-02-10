#!/usr/bin/env python
# coding: utf-8

# In[76]:


import cv2
import matplotlib.pyplot as plt


# In[77]:


config_file= r"C:\Users\91894\Downloads\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"


# In[78]:


frozen_model = r"C:\Users\91894\Downloads\frozen_inference_graph.pb"


# In[79]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[80]:


classLabels = []
file_name = r"C:\Users\91894\Downloads\labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[81]:


print(classLabels)


# In[82]:


print(len(classLabels))


# In[83]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127,5,127.5))
model.setInputSwapRB(True)


# In[84]:


img =cv2.imread(r"C:\Users\91894\Pictures\Saved Pictures\filip-mroz-zK049OFP4uI-unsplash.jpg")
plt.imshow(img)


# In[85]:


ClassIndex, confidence, bbox= model.detect(img, confThreshold= 0.5)


# In[86]:


print(ClassIndex)


# In[87]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes,(255,0,0),2)
    cv2.putText(img, classLabels[ClassInd-1],(boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=3)


# In[88]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[89]:


# # VIdeo
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cant open the video')

font_scale =3
font = cv2.FONT_HERSHEY_PLAIN
                       
while True:
    ret, frame = cap.read()
                       
    ClassIndex, confidece, bbox= model.detect(frame, confThreshold=0.55)
                       
    print(ClassIndex)
                       
    if len(ClassIndex)!=0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if ClassInd<=80:
                cv2.rectangle(frame, boxes,(255,0,0),2)
                cv2.putText(frame, classLabels[ClassInd-1],(boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=3)
    cv2.imshow('Obj detection', frame)
    
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
                       
cap.release()
cv2.destroyAllWindows()


# In[90]:


# import cv2

# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError('Cant open the video')

# font_scale = 3
# font = cv2.FONT_HERSHEY_PLAIN

# while True:
#     ret, frame = cap.read()

#     ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

#     print(ClassIndex)

#     if len(ClassIndex) != 0:
#         for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
#             if ClassInd <= 80:
#                 cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)
# #                 cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
#                             color=(0, 255, 0), thickness=3)
#     cv2.imshow('Obj detection', frame)

#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




