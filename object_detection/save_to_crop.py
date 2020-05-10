
# coding: utf-8

# In[1]:

import sys

import pickle
#with open('/home/brian/google-images-download/google_images_download/country_flag.pickle', 'rb') as handle:
#    OD_result = pickle.load(handle)
#with open('/home/brian/google-images-download/google_images_download/Germany_flag_real.pickle', 'rb') as handle:
#    OD_result = pickle.load(handle)
#with open('pickle/country_flag_all.pickle', 'rb') as handle:
#print "This is the name of the script: ", sys.argv[0]
flag_dir_name = sys.argv[1]
with open(sys.argv[2]+'.pickle', 'rb') as handle:
    OD_result = pickle.load(handle)


# In[2]:


for x,y in OD_result.items():
    print (x)
    break


# In[5]:


i=0
#get_ipython().run_line_magic('pylab', 'inline')
import matplotlib
matplotlib.use("Pdf")
from matplotlib import pyplot as plt
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

directory = flag_dir_name+'/'
if not os.path.exists(directory):
    os.makedirs(directory)
import cv2
bbox = {}
for x,y in OD_result.items():
    i+=1
    print (x)
    image = Image.open(x)
    #print image
  #print image_path
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.

    """
    vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          y['detection_boxes'],
          y['detection_classes'],
          y['detection_scores'],
          category_index,
          instance_masks=y.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
    """
    #print y['detection_boxes']
    #print y['detection_classes']
    data =x.split('/')[-1][:-4]
    #print data
    count = 0
    flag_n = 0
    for class_t in y['detection_classes']:
        
        if class_t == 85:
            #print y['detection_boxes'][count]
            cor = y['detection_boxes'][count]

            width, height = image.size
            #crop_img = img[int(width*cor[0]):int(width*cor[2]), int(height*cor[1]):int(height*cor[3])]
            area = (int(width*cor[1]),int(height*cor[0]), int(width*cor[3]),int(height*cor[2]))
            #area = (200,0,300,200) #l u r d
            #print area
            #print image.size
            cropped_img = image.crop(area)
            directory2 =directory+'/'+x.split('/')[-2]
            if not os.path.exists(directory2):
                os.makedirs(directory2)
            cropped_img.save(directory2+'/'+data+"_"+str(flag_n)+'.jpg')
            bbox[data+"_"+str(flag_n)] = area
            plt.show()
            flag_n+=1
        count+=1
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)
    #print x
    """
    img=mpimg.imread(x)
    imgplot = plt.imshow(img)
    plt.show()
    """
    #if i==5:
    #break


# In[6]:


#with open('pickle/bbox.pickle', 'wb') as f:
#    pickle.dump(bbox,f,protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


"""
from glob import glob
from collections import Counter
countryDic = Counter()
for name in glob('country_flag/*/*'):
    #print name
    data = name.split('/')[-2]
    countryDic[data] +=1
    #break
print countryDic
"""


# In[19]:


"""
count = 0
for x,y in countryDic.items():
    if y<500:
        count+=1
print count
"""

