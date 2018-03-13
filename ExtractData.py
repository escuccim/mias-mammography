
# coding: utf-8

# In[5]:


import tarfile

def extract_tar(fname, dest="./data/pgms"):
    mode = "r:gz" if (fname.endswith("tar.gz")) else "r:"
    tar = tarfile.open(fname, mode)
    tar.extractall(path=dest)
    tar.close()


# In[6]:


extract_tar("./data/all-mias.tar.gz")

