
# coding: utf-8

# In[87]:


import numpy as np
import re
from glob import glob
import pandas as pd
import pickle


# In[3]:


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
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


# In[4]:


# read all pgms in
files = glob('data/pgms/*.pgm')
data = []

for file in files:
    # read each file in and convert it to a float
    data.append(read_pgm(file) * 1.0)

images = np.array(data, dtype=np.float32)


# In[5]:


# save the images to an npy file so they don't need to be read in individually in the future
np.save('./images.npy', images)


# In[56]:


## Label Data


# In[102]:


# import and clean the annotation data
all_cases_df = pd.read_table('data/Info.txt', delimiter=' ')
all_cases_df = all_cases_df[all_cases_df.columns[:-1]] # drop last column
all_cases_df['path'] = all_cases_df['REFNUM'].map(lambda x: '%s.pgm' % x)


# In[103]:


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


# In[104]:


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


# In[105]:


# save the cleaned data
all_cases_df.to_pickle("all_cases_df.pkl")

np.save("labels.npy", class_names)

# clean up the annotations to remove unneeded columns
labels = all_cases_df.copy()
labels.drop(['CLASS','SEVERITY','path'], axis=1, inplace=True)

# fill radius with 0
labels['RADIUS'].fillna(0, inplace=True)
labels.to_pickle("labels.pkl")

