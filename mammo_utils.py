
# coding: utf-8

# In[16]:


import re
import numpy as np
import os
from PIL import Image, ImageMath
import shutil

## import PGM files and return a numpy array
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
        
    image = np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))
    
    image_id = int(re.findall('([\d]+)\.', filename)[0])
    
    if image_id % 2 != 0:
        image = np.fliplr(image)
        
    return image


# In[1]:


## flip every other image left to right so they are all oriented the same way
def read_images(file):
    # get the id of the images from the filename
    image_id = int(re.findall('([\d]+)\.', file)[0])
    print(image_id)
    # read in the image
    image = read_pgm(file)
    
    # if the ID is even flip the image
    if image_id % 2 != 0:
        image = np.fliplr(image)
        #image = X_return[...,::-1,:]
        
    return image


# In[2]:


## Extract a tar.gz file
def extract_tar(fname, dest="./data/pgms"):
    mode = "r:gz" if (fname.endswith("tar.gz")) else "r:"
    tar = tarfile.open(fname, mode)
    tar.extractall(path=dest)
    tar.close()


# In[ ]:


## Download files from a URL to a file in the data directory
def download_file(url, name):
    # check that the data directory exists
    try:
        os.stat("data")
    except:
        os.mkdir("data")  
    fname = wget.download(url, os.path.join('data',name)) 


# In[13]:


## flatten a directory tree to a single directory
def copy_subdirectories_to_directory(path, destination):
    subdirectories = os.listdir(path)
    for directory in subdirectories:
        files = os.listdir(os.path.join(path, directory))
        for file in files:
            shutil.copy(os.path.join(path, directory, file), destination)


# In[ ]:


## given a path to a directory containing images, open all files in the directory, convert the images to
## a numpy array, and return the array containing all the images, with the labels
def convert_images_to_array(path, label_data=None):
    data = []
    labels = []
    files = os.listdir(path)
    for file in files:
        img_data = Image.open(path + '/' + file)
        arr = np.array(img_data)
        
        if label_data is not None:
            # catch errors if there is no label for the image
            try:
                label = label_data.loc[file].CLASS
                labels.append(label)
                # if we have labels only add the image if there is a label
                data.append(arr)
            except:
                print("Error with", file)
                continue
        else:
            data.append(arr)
            
    return np.array(data), labels


# In[17]:


## Open a png image, convert it to a JPG and save it
def convert_png_to_jpg(path):
    # open the image
    im = Image.open(path)
    im2 = ImageMath.eval('im/256', {'im':im}).convert('L')
    
    # create a new file name
    path_array = path.split("\\")
    dir_path = "\\".join(path_array[:-1])
    image_name = path_array[-1]
    new_image_name = image_name.replace('.png','.jpg')
    new_image_name
    
    
    im2.convert('RGB').save(os.path.join(dir_path, new_image_name),"JPEG")


# In[ ]:


## The scans from one particular scanner (DBA) have white sections cut out of them, possibly to hide personal information
## this is only on the normal scans, so a convnet could use this information to identify the normal scans
## to prevent this I will replace all white pixels with black, as there are no pure white pixels in a normal scan

def remove_white_from_image(path, is_png=False):
    # open the image
    img = Image.open(path)
    
    if is_png:
        # convert to rgb
        img2 = ImageMath.eval('im/256', {'im':img}).convert('L')

    # turn the image into an array
    im_array = np.array(img2)
    
    # turn all white pixels to black
    im_array[im_array == 255] = 0
    
    return im_array


# In[ ]:


## Takes in a PIL image, resizes it to new_size square and returns the new images
def resize_image(img, new_size):
    img2 = img.resize((new_size, new_size), PIL.Image.ANTIALIAS)
    return img2

