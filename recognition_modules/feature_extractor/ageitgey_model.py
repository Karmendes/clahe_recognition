import face_recognition
import cv2

# Extract feature vector
def get_embeddings(rects,frame):
    """
    Gets encodings from the face

    Inputs: rects, frame
    Output: encoding
    """

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in [rects]]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # compute the facial embeddings for each face bounding box
    encoding = face_recognition.face_encodings(rgb, boxes)
    return encoding[0]