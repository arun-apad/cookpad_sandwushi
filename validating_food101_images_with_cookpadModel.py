
# coding: utf-8

# In[13]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import os
import cv2
print(cv2.__version__)

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from glob import glob


# In[14]:


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


# In[15]:


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


# In[16]:


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


# In[19]:


pathlist = Path("tf_files/food101/sushi").glob('**/*.j*')

i = 0
countCookpadonFood101Sushi = 0
model_file = "tf_files/cookpadModel/retrained_graph.pb"
label_file = "tf_files/cookpadModel/retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"
for path in pathlist:
    path_in_str = str(path)
    #print(path_in_str)
    im = Image.open(path_in_str)
    

    #print(path_in_str)
    
    
    file_name = path_in_str
    graph = load_graph(model_file)
    
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    
    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
        end=time.time()
        
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    #print(results.argsort()[-5:][::-1])
    i = i+1
    if results[1] > results[0]:
        countCookpadonFood101Sushi = countCookpadonFood101Sushi+1
    #print(labels[0], results[0])
    #print(labels[1], results[1])
    print("Validating Sushi Food101 Images "+str(countCookpadonFood101Sushi)+"  out of "+str(i)+" --- "+str(countCookpadonFood101Sushi/i))


# In[20]:


pathlist = Path("tf_files/food101/sandwich").glob('**/*.j*')

i = 0
countCookpadon101Sandwich = 0
model_file = "tf_files/cookpadModel/retrained_graph.pb"
label_file = "tf_files/cookpadModel/retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"
for path in pathlist:
    path_in_str = str(path)
    #print(path_in_str)
    im = Image.open(path_in_str)
    

    #print(path_in_str)
    
    
    file_name = path_in_str
    graph = load_graph(model_file)
    
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    
    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
        end=time.time()
        
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    #print(results.argsort()[-5:][::-1])
    i = i+1
    if results[0] > results[1]:
        countCookpadon101Sandwich = countCookpadon101Sandwich+1
    #print(labels[0], results[0])
    #print(labels[1], results[1])
    print("Validating Sandwich Food101 Images "+str(countCookpadon101Sandwich)+"  out of "+str(i)+" --- "+str(countCookpadon101Sandwich/i))

print("Images of Sushi from Food101 data = 1000")
print("Images of Sushi from Food101 data, that were correctly classified  = "+str(countCookpadonFood101Sushi))

print("Images of Sandwich from Food101 data = 1000")
print("Images of Sandwich from Food101 data, that were correctly classified  = "+str(countCookpadon101Sandwich))


# In[22]:


#Precision of the Model trained with Cookpad images and tested on Food101 images.
#(% of images classified as sushi was actually sushi)

Precision_sushi = countCookpadonFood101Sushi/(countCookpadonFood101Sushi + (1000 - countCookpadon101Sandwich))

print("(% of images classified as sushi was actually sushi)")
# Formula = No. images the model classified as sushi correctly / No. images the model classified as sushi instead of sandwich)
print("Formula = No. images the model classified as sushi correctly / No. images the model classified as sushi instead of sandwich")
print("Precision for Sushi by the Model trained with Cookpad images and tested on Food101 images.")
print(Precision_sushi)
print("\n")
print("------------------------------------------------------------------------------------------------------------------------------")


# In[23]:


#Precision of the Model trained with Cookpad images and tested on Food101 images.
#(% of images classified as sandwich was actually sandwich)
# Formula = No. images the model classified as sandwich correctly / No. images the model classified as sandwich instead of sandwich)

Precision_sandwich = countCookpadon101Sandwich/(countCookpadon101Sandwich + (1000 - countCookpadonFood101Sushi))
print("(% of images classified as sandwich was actually sandwich)")
print("Formula = No. images the model classified as sandwich correctly / No. images the model classified as sandwich instead of sandwich")
print("Precision for Sandwich by the Model trained with Cookpad images and tested on Food101 images.")
print(Precision_sandwich)
print("\n")
print("------------------------------------------------------------------------------------------------------------------------------")


# In[24]:


#Recall of the Model trained with Cookpad images and tested on Food101 images.
#(% of accuracy)
#Fomula = No. images the model classified as sushi correctly / total no of sushi images

Recall_sushi = countCookpadonFood101Sushi/1000 

print("(Accuracy %)")
print("Fomula = No. images the model classified as sushi correctly / total no of sushi images")
print("Recall for Sushi by the Model trained with Cookpad images and tested on Food101 images.")
print(Recall_sushi)
print("\n")
print("------------------------------------------------------------------------------------------------------------------------------")


# In[26]:


#Recall of the Model trained with Cookpad images and tested on Food101 images.
#(% of accuracy)
#Fomula = No. images the model classified as sanwich correctly / total no of sandwich images

Recall_sandwich = countCookpadon101Sandwich/1000 

print("(Accuracy %)")
print("Fomula = No. images the model classified as sushi correctly / total no of sushi images")
print("Recall for Sushi by the Model trained with Cookpad images and tested on Food101 images.")
print(Recall_sandwich)
print("\n")
print("------------------------------------------------------------------------------------------------------------------------------")

