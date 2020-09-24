import face_recognition


def get_name_election(encoding,data,input_tolerance = 0.5):
    """
    Matches the detected face to the known faces and returns the name of the recognized person

    Inputs: encoding, data, input_tolerance
    Output: name
    """

    # attempt to match each face in the input image to our encodings
    matches = face_recognition.compare_faces(data["encodings"],encoding, tolerance=input_tolerance)
    name = "Unknown"
    if True in matches:
    # find the indexes of all matched faces then initialize a
    # dictionary to count the total number of times each face
    # was matched
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
    
    return name