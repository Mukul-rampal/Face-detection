#!/usr/bin/env python
# coding: utf-8

# In[7]:


import mtcnn
print(mtcnn.__version__)
import cv2
import matplotlib.pyplot as plt
pic=plt.imread("D:/annivarsary/20190413080826_IMG_3426.jpg")
print(pic.shape)
plt.imshow(pic)
plt.show()
def facdetec(filename, result_list):
    data=plt.imread(filename)
    plt.imshow(data)
    ax=plt.gca()
    for result in result_list:
        x,y,width,height=result['box']
        rect=plt.Rectangle((x,y),width,height,fill=False,color='green')
        ax.add_patch(rect)
    plt.show()
file="D:/annivarsary/20190413080826_IMG_3426.jpg"
pic=plt.imread(file)
# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# bboxes = classifier.detectMultiScale(pixels)
detector=mtcnn.MTCNN()
faces=detector.detect_faces(pic)
facdetec(file,faces)    


# In[ ]:




