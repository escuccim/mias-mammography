
# coding: utf-8

# In[1]:


import re
import numpy as np
import os
from PIL import Image, ImageMath
from scipy.misc import imresize
import shutil
import matplotlib.pyplot as plt
import PIL
import sys

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
                    
    # make sure the destination directory exists
    try:
        os.stat(destination_path)
    except:
        os.mkdir(destination_path)  
    
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
                new_name = patient_id+'_'+image_side+'_'+image_type+'.jpg'
                
                # if the file already exists in the final destination, rename it
                if os.path.exists(destination_path + new_name):
                    new_name = patient_id+'_'+image_side+'_'+image_type+'_1.jpg'
                
                new_path = path + new_name
                os.rename(path+'/'+file, new_path)
            except:
                continue
            
            ## copy the files so they are all in one directory
            shutil.copy(new_path, destination_path)

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
def create_mask(mask_path, full_image_arr, slice_size=598, return_size=False, half=True, output=True):
    # open the mask
    mask = PIL.Image.open(mask_path)
    
    # cut the image in half
    if half:
        h, w = mask.size
        new_size = ( h // 2, w // 2)
        mask.thumbnail(new_size, PIL.Image.ANTIALIAS)

    # turn it into an arry
    mask_arr = np.array(mask)
    
    # get rid of the extras dimensions
    mask_arr = mask_arr[:,:,0]
    
    # some images have white on the borders which may be something a convnet can use to predict. To prevent this,
    # if the full image has more than 50,000 white pixels we will trim the edges by 20 pixels on either side
    if np.sum(np.sum(full_image_arr >= 225)) > 20000:
        full_image_arr = remove_margins(full_image_arr)
        mask_arr = remove_margins(mask_arr)
        if output:
            print("Trimming borders", mask_path)
            
    # make sure the mask is the same size as the full image, if not there is a problem, don't use this one
    if mask_arr.shape != full_image_arr.shape:
        # see if the ratios are the same
        mask_ratio = mask_arr.shape[0] / mask_arr.shape[1]
        image_ratio = full_image_arr.shape[0] / full_image_arr.shape[1]
        
        if abs(mask_ratio - image_ratio) <=  1e-03:
            if output:
                print("Mishaped mask, resizing mask", mask_path)
            
            # reshape the mask to match the image
            mask_arr = imresize(mask_arr, full_image_arr.shape)
            
        else:
            if output:
                print("Mask shape:", mask_arr.shape)
                print("Image shape:", full_image_arr.shape)
            print("Mask shape doesn't match image!", mask_path)
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
    
    mask_size = [row_size, col_size]
    
    # signal if the mask is bigger than the slice
    too_big = False
    
    if (last_col - first_col > slice_size + 30) or (last_row - first_row > slice_size + 30):
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
def create_slices(path, output=True, var_upper_threshhold=0, var_lower_threshhold=0, mean_threshold=0, stride=200):
    files = os.listdir(path)
    normal_slices = []
    normal_labels = []
    
    i = 0
    for file in files:
        if output:
            print(i, "-", file)
        i += 1
        tiles = slice_normal_image(os.path.join(path, file), var_upper_threshold=var_upper_threshhold, var_lower_threshold=var_lower_threshhold, mean_threshold=mean_threshold, stride=stride)
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
def slice_normal_image(path, var_upper_threshold=0, var_lower_threshold=0, mean_threshold=0, stride=200):
    # load the image
    img = PIL.Image.open(path)
    
    # convert the image to RGB
    img = PIL.ImageMath.eval('im/256', {'im':img}).convert('L')
    
    # size the image down by a random factor for variety and to try to match what we did to the cbis images
    scale_by = np.random.uniform(low=1.8, high=3.2)
    h, w = img.size
    new_size = ( h // scale_by, w // scale_by)
    img.thumbnail(new_size, PIL.Image.ANTIALIAS)
    
    # convert to an array
    img = np.array(img)
    
    # remove white pixels
    img[img == 255] = 0
    
    # remove 7% from each side of image to eliminate borders
    h, w = img.shape
    hmargin = int(h * 0.07)
    wmargin = int(w * 0.07)
    img = img[hmargin:h-hmargin, wmargin:w-wmargin]
    
    # slice the image into 299x299 tiles
    size = 299
    tiles = [img[x:x+size,y:y+size] for x in range(0,img.shape[0],stride) for y in range(0,img.shape[1],stride)]
    usable_tiles = []
    
    # for each tile:
    for i in range(len(tiles)):
        # make sure tile has correct shape
        if tiles[i].shape == (size,size):
            # make sure the tile doesn't have too many white or black pixels, that indicates it is not useful
            if (np.sum(np.sum(tiles[i] >= 225)) < 100) and (np.sum(np.sum(tiles[i] <= 20)) <= 50000):
                # make sure tile has stuff in it
                if np.mean(tiles[i]) >= mean_threshold:
                    # make sure the tile contains image and not mostly empty space
                    if np.var(tiles[i]) <= var_upper_threshold:
                        if np.var(tiles[i]) >= var_lower_threshold:
                            # reshape the tile so they will work with the convnet
                            usable_tiles.append(random_flip_image(tiles[i].reshape(299,299,1)))

    return usable_tiles


# In[ ]:


## create a random offset for slices to have some variety in the data
def get_fuzzy_offset(roi_size, slice_size=299):
    fuzz_factor = (slice_size - roi_size - 20) // 2
    
    if fuzz_factor <= 0:
        fuzz_factor = 1
    
    fuzz_sign_h = np.random.binomial(1,0.5)
    fuzz_sign_w = np.random.binomial(1,0.5)

    fuzz_offset_w = np.random.randint(low=0, high=fuzz_factor)
    if fuzz_sign_w == 0:
        fuzz_offset_w = 0 - fuzz_offset_w

    fuzz_offset_h = np.random.randint(low=0, high=fuzz_factor)
    if fuzz_sign_h == 0:
        fuzz_offset_h = 0 - fuzz_offset_h
    
    return fuzz_offset_h, fuzz_offset_w


# In[ ]:


## Progress bar taken from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


# In[ ]:


def get_roi_edges(center_col, center_row, img_height, img_width, fuzz_offset_w=0, fuzz_offset_h=0, scale_factor=1, slice_size=299):
    # slice margin
    slice_margin = slice_size // 2
    
    # figure out the new center of the ROI
    center_col_scaled = int(center_col * scale_factor)
    center_row_scaled = int(center_row * scale_factor)
    
    start_col = int(center_col_scaled - slice_margin + fuzz_offset_h)
    end_col = int(start_col + slice_size)
    
    if start_col < 0:
        start_col = 0
        end_col = slice_size
    elif end_col > img_width:
        end_col = img_width
        start_col = int(img_width - slice_size)
        
    start_row = int(center_row_scaled - slice_margin + fuzz_offset_w)
    end_row = int(start_row + slice_size)
    
    if start_row < 0:
        start_row = 0
        end_row = slice_size
    elif end_row > img_height:
        end_row = img_height
        start_row = int(img_height - slice_size)
     
    return start_row, end_row, start_col, end_col


# In[ ]:


## randomly rotate an image
def random_rotate_image(img):
    rotations = np.random.randint(low=-3, high=3)
    return np.rot90(img, rotations)

## randomly flip an image left-right, up-down or both and return it
def random_flip_image(img):
    fliplr = np.random.binomial(1,0.5)
    flipud = np.random.binomial(1,0.5)
    
    if fliplr:
        img = np.flip(img, 1)
    if flipud:
        img = np.flip(img, 0)
        
    return random_rotate_image(img)


# In[ ]:


## cut out tiles from images given the ROI center and size, with padding, random offset and random rotation
def extract_slice(img, center_col, center_row, roi_size, padding=1.2, context_scale=2, return_slice_size=299, distort=True):
    # figure out the size of the tile we will extract
    tile_size = int(roi_size * context_scale)
    
    # if the tile is very small enlarge it so we don't zoom too much and distort the images
    if tile_size < 300:
        tile_size = 300
    
    # get random offset
    fuzz_offset_h, fuzz_offset_w = get_fuzzy_offset(int(roi_size * padding), slice_size=tile_size)
    
    # define boundaries for the abnormality
    image_h = img.shape[0]
    image_w = img.shape[1]
    start_row, end_row, start_col, end_col = get_roi_edges(center_col, center_row, image_h, image_w, fuzz_offset_w, fuzz_offset_h, 1, slice_size=tile_size)
    
    # slice the ROI out of the image
    img_slice = img[start_row:end_row, start_col:end_col]

    # cut the slice down to proper size
    try:
        img_slice = imresize(img_slice, (return_slice_size,return_slice_size))
    except:
        print("Error resizing tile")
        return np.array([1,1])
    
    # if everything is usable return it, otherwise return an unusable slice
    if img_slice.shape == (return_slice_size,return_slice_size):
        if distort:
            return random_flip_image(img_slice.reshape(return_slice_size,return_slice_size,1))
        else:
            return img_slice.reshape(return_slice_size,return_slice_size,1)
    else:
        return np.array([1,1])


# In[ ]:


## remove extraneous characters from end of file name and return it
def clean_name(name):
    patient_id = re.findall("(P_[\d]+)_", name)
    if len(patient_id) > 0:
        patient_id = patient_id[0]
    else:
        print("Name error")
        return name

    image_side = re.findall("_(LEFT|RIGHT)_", name)

    if len(image_side) > 0:
        image_side = image_side[0]
    else:
        print("Side error")
        return name

    image_type = re.findall("(CC|MLO)", name)
    if len(image_type) > 0:
        image_type = image_type[0]
    else:
        return name
    
    return patient_id + "_" + image_side + "_" + image_type


# In[ ]:


def plot_metrics(model_name, classification, dataset):
    train_acc_values = np.load(os.path.join("data", "results", model_name, model_name + "train_acc.npy"))
    valid_acc_values = np.load(os.path.join("data", "results", model_name, model_name + "cv_acc.npy"))

    train_cost_values = np.load(os.path.join("data", "results", model_name, model_name + "train_loss.npy"))
    valid_cost_values = np.load(os.path.join("data", "results", model_name, model_name + "cv_loss.npy"))

    train_recall_values = np.load(os.path.join("data", "results", model_name, model_name + "train_recall.npy"))
    valid_recall_values = np.load(os.path.join("data", "results", model_name, model_name + "cv_recall.npy"))

    train_lr_values = np.load(os.path.join("data", "results", model_name, model_name + "train_lr.npy"))
    
    # initialize the plots
    f, ax = plt.subplots(1, 4, figsize=(24, 5))

    ax[0].plot(valid_acc_values, color="red", label="Validation")
    ax[0].plot(train_acc_values, color="blue", label="Training")
    ax[0].axhline(y=0.83, color="salmon", label="Baseline Accuracy")
    ax[0].set_title('Validation accuracy: {:.4f} (mean last 4)'.format(np.mean(valid_acc_values[-4:])))
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim([0.4,1.0])
    ax[0].legend()

    ax[1].plot(valid_cost_values, color="red", label="Validation")
    ax[1].plot(train_cost_values, color="blue", label="Training")
    ax[1].set_title('Validation x-entropy: {:.3f} (mean last 4)'.format(np.mean(valid_cost_values[-4:])))
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Cross Entropy')
    ax[1].set_ylim([0,2.0])
    ax[1].legend()

    ax[2].plot(train_recall_values, color="red", label="Validation")
    ax[2].plot(valid_recall_values, color="blue", label="Training")
    ax[2].set_title('Validation Recall: {:.3f} (mean last 4)'.format(np.mean(valid_recall_values[-4:])))
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Recall')
    ax[2].legend()

    ax[3].plot(train_lr_values)
    ax[3].set_title("Learning rate: {:.6f}".format(np.mean(train_lr_values[-1:])))
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Learning Rate")

    f.suptitle("Results for " + model_name + " " + classification + " (Dataset " + str(dataset) + ")")
    plt.show()

