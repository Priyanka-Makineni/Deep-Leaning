#!/usr/bin/env python
# coding: utf-8

# # Object Detection

# ## Definition

# Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class
# (such as humans, buildings,cars etc) in digital images and videos.
# 

# ### FOLLOWING PACKAGES NEED TO BE INSTALLED :¶
# Tensorflow/Numpy/SciPy/OpenCV/Pillow/Matplotlib/H5py/Keras/ImageAI
# 
# We’ll be using a trained RetinaNet computer vision model to perform the detection and recognition tasks.This model is trained to detect and recognize 80 different objects, named below:
# 
# person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop_sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donot, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair dryer, toothbrush.
# 
# Copy the RetinaNet model file and the image you want to detect to the folder that contains the python file.
# 
#     "**To download Click on [RetinaNet](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5)**"
# 
# 

# In[1]:


from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()


# In the above 3 lines, we imported the ImageAI object detection class in the first line, imported the python os class in the second line and defined a variable to hold the path to the folder where our python file, RetinaNet model file and images are in the third line.
# 
# 

# In[2]:


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "market.jpg"), output_image_path=os.path.join(execution_path , "market1.jpg"))


# In the 5 lines of code above, we defined our object detection class in the first line, set the model type to RetinaNet in the second line, set the model path to the path of our RetinaNet model in the third line, load the model into the object detection class in the fourth line, then we called the detection function and parsed in the input image path and the output image path in the fifth line.
# 
# 

# In[3]:


for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


# In the above 2 lines of code, we iterate over all the results returned by the detector.detectObjectsFromImage function in the first line, then print out the name and percentage probability of the model on each object detected in the image in the second line.
# 
# 

# In[4]:


detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "market.jpg"), output_image_path=os.path.join(execution_path , "diff images market.jpg"), extract_detected_objects=True)


# ImageAI supports many powerful customization of the object detection process. One of it is the ability to extract the image of each object detected in the image. By simply parsing the extra parameter extract_detected_objects=True into the detectObjectsFromImage function as seen below, the object detection class will create a folder for the image objects, extract each image, save each to the new folder created and return an extra array that contains the path to each of the images.
# 
# ImageAI provides many more features useful for customization and production capable deployments for object detection tasks. Some of the features supported are:
# 
# * Adjusting Minimum Probability: By default, objects detected with a probability percentage of less than 50 will not be shown or reported. You can increase this value for high certainty cases or reduce the value for cases where all possible objects are needed to be detected.
# * Custom Objects Detection: Using a provided CustomObject class, you can tell the detection class to report detections on one or a few number of unique objects.
# * Detection Speeds: You can reduce the time it takes to detect an image by setting the speed of detection speed to “fast”, “faster” and “fastest”.
# * Input Types: You can specify and parse in file path to an image, Numpy array or file stream of an image as the input image.
# * Output Types: You can specify that the detectObjectsFromImage function should return the image in the form of a file or Numpy array.
# 

# In[ ]:




