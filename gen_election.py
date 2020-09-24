# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from imutils import paths
import cv2
import os
import pickle
from recognition_modules.pre_process.pre_process import pre_process_frame
from recognition_modules.feature_extractor.ageitgey_model import get_embeddings
from recognition_modules.face_match.election_based import get_name_election


# %%
with open('encodings.pickle', 'rb') as handle:
    data = pickle.load(handle)


# %%
# load paths
paths = list(paths.list_images("Train/"))
# load detector
detector = cv2.CascadeClassifier('detector.xml')


# %%
results = {
    'predict':[],
    'real':[],
    'tolerance':[]
}


# %%
for i in range(50,51):
    print(f'Info: Tolerancia {i/100}')
    for path in paths:
        try:
            # get the frame
            frame = cv2.imread(path)
            # pre process frame
            rects = pre_process_frame(frame,detector)
            # get embeddings
            encoding = get_embeddings(rects,frame)
            # get tha name
            name = get_name_election(encoding,data,input_tolerance= (i/100))
            # save in dictionary
            results['predict'].append(name)
            results['real'].append(path.split(os.path.sep)[-2])
            results['tolerance'].append((i/100))
        except:
            print(f'Info: Tolerancia {i/100}')


# %%
with open('baseline.pickle', 'wb') as handle:
    pickle.dump(results, handle)


