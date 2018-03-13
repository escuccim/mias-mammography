
# coding: utf-8

# In[1]:


import numpy as np
import re
from glob import glob
import pandas as pd
import pickle
import os
import re
from mammo_utils import read_pgm


# In[64]:


# read all pgms in
files = glob('data/pgms/*.pgm')
data = []

for file in files:
    # read each file in and convert it to a float
    data.append(read_pgm(file) * 1.0)
    
images = np.array(data, dtype=np.int16)

# save the images to an npy file so they don't need to be read in individually in the future
np.save(os.path.join('data','images.npy'), images)


# In[2]:


## do same for small resized pgms
files = glob('data/small/*.pbm')
small_data = []

for file in files:
    # read each file in and convert it to a float
    small_data.append(read_pgm(file) * 1.0)

small_images = np.array(small_data, dtype=np.int16)
np.save(os.path.join('data','small_images.npy'), small_images)


# In[67]:


## And again for medium sized pgms
files = glob('data/medium/*.pbm')
med_data = []

for file in files:
    # read each file in and convert it to a float
    med_data.append(read_pgm(file) * 1.0)

medium_images = np.array(med_data, dtype=np.int16)
np.save(os.path.join('data','medium_images.npy'), medium_images)


# In[5]:


## Label Data


# In[6]:


# import and clean the annotation data
all_cases_df = pd.read_table('data/Info.txt', delimiter=' ')
all_cases_df = all_cases_df[all_cases_df.columns[:-1]] # drop last column
all_cases_df['path'] = all_cases_df['REFNUM'].map(lambda x: '%s.pgm' % x)


# In[7]:


# Let's drop the duplicate rows
all_cases_df.drop_duplicates(subset=['REFNUM'], keep='first', inplace=True)

# reindex it
all_cases_df.reset_index(inplace=True, drop=True)

# add another column to indicate whether the scan is normal or not, disregarding the specific type of abnormality
all_cases_df['ABNORMAL'] = (all_cases_df['CLASS'] != 'NORM') * 1.0

# add one more column to indicate whether the abnormality is benign or malignant, malignant is 1, benign is 0, no abnormality is -1
all_cases_df['TYPE'] = -1
mal_idx = all_cases_df['SEVERITY'] == 'M'
all_cases_df.loc[mal_idx,'TYPE'] = 1
ben_idx = all_cases_df['SEVERITY'] == 'B'
all_cases_df.loc[ben_idx,'TYPE'] = 0


# In[8]:


from sklearn.preprocessing import LabelEncoder

# encode the classes
class_le = LabelEncoder()
class_le.fit(all_cases_df.CLASS)
class_vec = class_le.transform(all_cases_df.CLASS)

# save the class names to a vector and add the encoded classes to the df
class_names = class_le.classes_
all_cases_df['CLASS_Y'] = class_vec

# one hot encode the backgrounds
all_cases_df = pd.get_dummies(all_cases_df, columns=['BG'])


# In[11]:


# save the cleaned data
all_cases_df.to_pickle(os.path.join("data","all_cases_df.pkl"))

np.save(os.path.join("data","names.npy"), class_names)

# clean up the annotations to remove unneeded columns
labels = all_cases_df.copy()
labels.drop(['CLASS','SEVERITY','path'], axis=1, inplace=True)

# fill radius with 0
labels['RADIUS'].fillna(0, inplace=True)
labels.to_pickle(os.path.join("data","labels.pkl"))

