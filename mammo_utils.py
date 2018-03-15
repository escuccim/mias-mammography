
# coding: utf-8

# In[1]:


import re
import numpy as np
import os
from PIL import Image

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


# flip every other image left to right so they are all oriented the same way
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


def extract_tar(fname, dest="./data/pgms"):
    mode = "r:gz" if (fname.endswith("tar.gz")) else "r:"
    tar = tarfile.open(fname, mode)
    tar.extractall(path=dest)
    tar.close()


# In[ ]:


def download_file(url, name):
    # check that the data directory exists
    try:
        os.stat("data")
    except:
        os.mkdir("data")  
    fname = wget.download(url, os.path.join('data',name)) 


# In[ ]:


def convert_images_to_array(path, label_data=None):
    data = []
    labels = []
    files = os.listdir(path)
    for file in files:
        img_data = Image.open(path + '/' + file)
        arr = np.array(img_data)
        data.append(arr)
        
        if label_data is not None:
            label = label_data.loc[file].CLASS
            labels.append(label)
            
    return np.array(data), labels

