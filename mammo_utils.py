
# coding: utf-8

# In[1]:


import re
import numpy as np
import os
from PIL import Image, ImageMath
from scipy.misc import imresize
import shutil
import PIL

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
    
    return image


# In[2]:


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


# In[3]:


## Extract a tar.gz file
def extract_tar(fname, dest="./data/pgms"):
    mode = "r:gz" if (fname.endswith("tar.gz")) else "r:"
    tar = tarfile.open(fname, mode)
    tar.extractall(path=dest)
    tar.close()


# In[4]:


## Download files from a URL to a file in the data directory
def download_file(url, name):
    # check that the data directory exists
    try:
        os.stat("data")
    except:
        os.mkdir("data")  
    fname = wget.download(url, os.path.join('data',name)) 


# In[5]:


## flatten a directory tree to a single directory
def copy_subdirectories_to_directory(path, destination):
    subdirectories = os.listdir(path)
    for directory in subdirectories:
        files = os.listdir(os.path.join(path, directory))
        for file in files:
            shutil.copy(os.path.join(path, directory, file), destination)


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


## Takes in a PIL image, resizes it to new_size square and returns the new images
def resize_image(img, new_size):
    img2 = img.resize((new_size, new_size), PIL.Image.ANTIALIAS)
    return img2


# In[11]:


## rename the files to include the patient id so we can match the image up with the labels
## also copy the images to a single directory so we have them all in one place
def rename_and_copy_files(path, sourcedir="JPEG512", destdir="AllJPEGS512"):
    directories = os.listdir(path+sourcedir)
    source_path = path + sourcedir + "/"
    destination_path = path + destdir + "/"
    
    # keep a counter so each file has a unique name
    i = 1
    
    # loop through the directories
    for directory in directories:
        # get the patient number and image type from the directory name
        patient_id = str(re.findall("_(P_[\d]+)_", directory))
        if len(patient_id) > 0:
            patient_id = patient_id[0]
        else:
            continue
            
        image_side = str(re.findall("_(LEFT|RIGHT)_", directory))
        
        if len(image_side) > 0:
            image_side = image_side[0]
        else:
            continue
        
        image_type = str(re.findall("(CC|MLO)", directory))
        if len(image_type) > 0:
            image_type = image_type[0]
        else:
            continue
        
        if not patient_id:
            continue
            
        # get the subdirectories
        subdir = os.listdir(source_path+directory)

        # get the next level of subdirectories
        subsubdir = os.listdir(source_path+directory+'/'+subdir[0])

        # get the files 
        files = os.listdir(source_path+directory+'/'+subdir[0]+'/'+subsubdir[0])
        path = source_path+directory+'/'+subdir[0]+'/'+subsubdir[0]

        for file in files:
            # rename the file so we know where it came from
            # some of the data is not properly labeled, if that is the case skip it since we won't be able to label it
            try:
                new_name = path+'/'+patient_id+'_'+image_side+'_'+image_type+'.jpg'
                os.rename(path+'/'+file, new_name)
            except:
                continue
                
            # make sure the destination directory exists
            try:
                os.stat(destination_path)
            except:
                os.mkdir(destination_path)  
            
            # copy the files so they are all in one directory
            shutil.copy(new_name, destination_path)

        i += 1


# In[ ]:


## function to trim pixels off all sides of an image, should help to remove white borders and such
def remove_margins(image_arr, margin=20):
    h, w = image_arr.shape
    
    new_image = image_arr[margin:h-margin,margin:w-margin]
    
    return new_image


# In[12]:


## input: path to mask image PNG
## opens the mask, reduces its size by half, finds the borders of the mask and returns the center of the mass
## if the mass is bigger than the slice it returns the upper left and lower right corners of the mask as tuples
## which will be used to create multiple slices
## returns: center_row - int with center row of mask, or tuple with edges of the mask if the mask is bigger than the slice
##          center_col - idem
##          too_big - boolean indicating if the mask is bigger than the slice
def create_mask(mask_path, full_image_arr, slice_size=598, return_size=False):
    # open the mask
    mask = PIL.Image.open(mask_path)
    
    # cut the image in half
    h, w = mask.size
    new_size = ( h // 2, w // 2)
    mask.thumbnail(new_size, PIL.Image.ANTIALIAS)

    # turn it into an arry
    mask_arr = np.array(mask)
    
    # get rid of the extras dimensions
    mask_arr = mask_arr[:,:,0]
    
    # some images have white on the borders which may be something a convnet can use to predict. To prevent this,
    # if the full image has more than 50,000 white pixels we will trim the edges by 20 pixels on either side
    if np.sum(np.sum(full_image_arr >= 245)) > 20000:
        full_image_arr = remove_margins(full_image_arr)
        mask_arr = remove_margins(mask_arr)
        print("Trimming borders", mask_path)
        
    # make sure the mask is the same size as the full image, if not there is a problem, don't use this one
    if mask_arr.shape != full_image_arr.shape:
        # see if the ratios are the same
        mask_ratio = mask_arr.shape[0] / mask_arr.shape[1]
        image_ratio = full_image_arr.shape[0] / full_image_arr.shape[1]
        
        if abs(mask_ratio - image_ratio) <=  1e-03:
            print("Mishaped mask, resizing mask", mask_path)
            
            # reshape the mask to match the image
            mask_arr = imresize(mask_arr, full_image_arr.shape)
            
        else:
            print("Mask shape doesn't match image!", mask_path)
            print("Mask shape:", mask_arr.shape)
            print("Image shape:", full_image_arr.shape)
            return 0, 0, False, full_image_arr, 0
    
    # find the borders
    mask_mask = mask_arr == 255

    # does each row or column have a white pixel in it?
    cols = np.sum(mask_mask, axis=0)
    rows = np.sum(mask_mask, axis=1)

    # figure out where the corners are
    first_col = np.argmax(cols > 0)
    last_col = mask_arr.shape[1] - np.argmax(np.flip(cols, axis=0) > 0)
    center_col = int((first_col + last_col) / 2)

    first_row = np.argmax(rows > 0)
    last_row = mask_arr.shape[0] - np.argmax(np.flip(rows, axis=0) > 0)
    center_row = int((first_row + last_row) / 2)
    
    col_size = last_col - first_col
    row_size = last_row - first_row
    
    mask_size = np.max([col_size, row_size])
    
    # signal if the mask is bigger than the slice
    too_big = False
    if (last_col - first_col > slice_size + 10) or (last_row - first_row > slice_size + 10):
        # since it seems that masses are best defined by their edges, if the mask is bigger than the slice
        # return tuples defining the corners of the mask. We will use those to create two slices containing 
        # borders for the mass - and give a 10 pixel border on each side so we can see the edges
        center_col = (first_col - 10, last_col + 10)
        center_row = (first_row - 10, last_row + 10)
        too_big = True
    
  
    return center_row, center_col, too_big, full_image_arr, mask_size


# In[ ]:


## takes a PIL image as input, scales the image to half size and returns it
def half_image(image):
    h,w = image.size
    
    new_size = ( h // 2, w // 2)
    image.thumbnail(new_size, PIL.Image.ANTIALIAS)
    
    return image


# In[ ]:


## function to read images contained in a directory, create slices from them and return a numpy array of the slices
## with labels. The threshholds are used to filter out images which are not usable or interesting. 
def create_slices(path, output=True, var_upper_threshhold=0, var_lower_threshhold=0, mean_threshold=0):
    files = os.listdir(path)
    normal_slices = []
    normal_labels = []
    
    i = 0
    for file in files:
        if output:
            print(i, "-", file)
        i += 1
        tiles = slice_normal_image(os.path.join(path, file), var_upper_threshold=var_upper_threshhold, var_lower_threshold=var_lower_threshhold, mean_threshold=mean_threshold)
        for tile in tiles:
            normal_slices.append(tile)
            normal_labels.append("NORMAL")
        
        #if i > 400:
        #    break
            
    assert(len(normal_slices) == len(normal_labels))
    
    return np.array(normal_slices), np.array(normal_labels)


# In[10]:


## Loads a PNG image, converts it to an RGB numpy array, slices it into tiles and returns the tiles which contain usable images
## Var and Mean thresholds can be used to only keep images with usable information in them.
## Inputs: path - path to image
##         var_threshold - only keep images with a variance BELOW this
##         mean_threshold - only keep images with a mean ABOVE this

def slice_normal_image(path, var_upper_threshold=0, var_lower_threshold=0, mean_threshold=0):
    # load the image
    img = PIL.Image.open(path)
    
    # convert the image to RGB
    img = PIL.ImageMath.eval('im/256', {'im':img}).convert('L')
    
    # size the image down by half
    h, w = img.size
    new_size = ( h // 2, w // 2)
    img.thumbnail(new_size, PIL.Image.ANTIALIAS)
    
    # convert to an array
    img = np.array(img)
    
    # remove white pixels, this will prevent images which contain text or white patches from being used
    # img[img > 225] = 0
    
    # remove 7% from each side of image to eliminate borders
    h, w = img.shape
    hmargin = int(h * 0.07)
    wmargin = int(w * 0.07)
    img = img[hmargin:h-hmargin, wmargin:w-wmargin]
    
    # slice the image into 299x299 tiles
    size = 299
    tiles = [img[x:x+size,y:y+size] for x in range(0,img.shape[0],size) for y in range(0,img.shape[1],size)]
    usable_tiles = []
    
    # for each tile:
    for i in range(len(tiles)):
        # make sure tile has correct shape
        if tiles[i].shape == (size,size):
            if np.sum(np.sum(tiles[i] >= 225)) < 100:
                # make sure tile has stuff in it
                if np.mean(tiles[i]) >= mean_threshold:
                    # make sure the tile contains image and not mostly empty space
                    if np.var(tiles[i]) <= var_upper_threshold:
                        if np.var(tiles[i]) >= var_lower_threshold:
                            # reshape the tile so they will work with the convnet
                            usable_tiles.append(tiles[i].reshape(299,299,1))

    return usable_tiles

