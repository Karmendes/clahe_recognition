

# import joblib
from joblib import load
import os

# Import knn model
knn = load('/Users/lucas/face_recognition/recognition/recog/face_match/knn_based/knn_office.joblib') 

# Get the result from predict
def get_name_knn(encoding,threshold,knn):
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn.kneighbors(encoding, n_neighbors=1)
    
    if closest_distances[0][0][0] < threshold:
        return knn.predict(encoding)[0]
    else:
        return 'Unknown'

    
    