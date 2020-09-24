# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from imutils import paths
import cv2
import os
from recognition_modules.pre_process.pre_process import pre_process_frame
from recognition_modules.feature_extractor.ageitgey_model import get_embeddings
from recognition_modules.face_match.election_based import get_name_election


# %%
# load paths
paths_encodings = list(paths.list_images("Test/"))
# load detector
detector = cv2.CascadeClassifier('detector.xml')


# %%
# set dataset 
data = {
    'name':[],
    'encodings':[]
}


# %%
# get the frame
frame = cv2.imread(paths_encodings[0])
# pre process frame
rects = pre_process_frame(frame,detector)
# get embeddings
encoding = get_embeddings(rects,frame)


# %%
for path in paths_encodings:
    try:
        # get the frame
        frame = cv2.imread(path)
        # pre process frame
        rects = pre_process_frame(frame,detector)
        # get embeddings
        encoding = get_embeddings(rects,frame)
        # save in the dictionary
        data['name'].append(path.split(os.path.sep)[-2])
        data['encodings'].append(encoding)
    except:
        print(f'{path} com problema')


# %%
import pandas as pd
data = pd.DataFrame(data)
length = data.groupby('name')['name'].agg('count')
data_count = pd.DataFrame(
    {
        'name':length.index.to_list(),
        'tamanho':length.to_list()
    }
)
data_final = data.set_index('name').join(data_count.set_index('name'))
data_final = data_final[data_final['tamanho'] >= 2]
data_final = data_final.reset_index()
data_final = data_final[['name','encodings']]
data = {
    'name':data_final['name'].to_list(),
    'encodings':data_final['encodings'].to_list()
}


# %%
import pickle
with open('encodings.pickle', 'wb') as handle:
    pickle.dump(data, handle)


