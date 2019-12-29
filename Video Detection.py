#!/usr/bin/env python
# coding: utf-8

# ### Object Detection Video with TensorFlow

# ##### What is Object Detection?

# Object detection is a computer vision technique that works to identify and locate objects within an image or video. Specifically, object detection draws bounding boxes around these detected objects, which allow us to locate where said objects are in (or how they move through) a given scene.

# Image recognition assigns a label to an image. A picture of a dog receives the label “dog”. A picture of two dogs, still receives the label “dog”. Object detection, on the other hand, draws a box around each dog and labels the box “dog”. The model predicts where each object is and what label should be applied. In that way, object detection provides more information about an image than recognition.

# To start performing **Video Object Detection**, you must download the **RetinaNet, YOLOv3 or TinyYOLOv3** object detection model via the links below:
# 
# **RetinaNet** (Size = 145 mb, high performance and accuracy, with longer detection time)<br>
# **YOLOv3** (Size = 237 mb, moderate performance and accuracy, with a moderate detection time)<br>
# **TinyYOLOv3** (Size = 34 mb, optimized for speed and moderate performance, with fast detection time)<br>
# 
# We can **download** the model file by using below link.
# 
# [YOLO.H5](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5)<br>
# [YOLO.tiny.H5](https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0/)<br>
# [RetinaNet](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5)
# 

# These models supported by ImageAI can detect 80 different types of objects. They are:<br>
# 
#       person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop_sign,
#       parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,
#       giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,
#       sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,
#       bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,   orange,
#       broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,
#       dining table,   toilet,   tv,   laptop,   mouse,   remote,   keyboard,   cell phone,   microwave,
#       oven,   toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer,
#       toothbrush.

# ###### For processing the video detection the following packaes need to be installed on your pc
# TensorFlow/Numpy/SciPy/OpenCV/Pillow/Matplotlib/h5py/keras/ImageAI

# In[2]:


# User defined function for printing output
def forFrame(frame_number, output_array, output_count, detected_copy):
    print("FOR FRAME " , frame_number)
#    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")


# In[3]:


from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "E:\\my data\\DATA SCIENCE\\DEEP LERNING\\Input\\Videos\\traffic-mini.mp4"),
                                output_file_path=os.path.join(execution_path, "E:\\my data\\DATA SCIENCE\\DEEP LERNING\\Output\\video\\traffic-mini1.mp4")
                                , frames_per_second=20,
                                             per_frame_function = forFrame,
                                             minimum_percentage_probability=30,
                                             log_progress=True,
                                             return_detected_frame=True)

print(video_path)


# In[ ]:




