import os
import face_recognition
import cv2
import matplotlib.pyplot as plt
# Make a list of available images in the "WhoAreThese" directory
images_to_be_matched = os.listdir('WhoAreThese')

# Make a list of available images in the "knownImages" directory
known_images = os.listdir('knownImages')
#check4image = "WhoIsJ.jpg"

for check4image in images_to_be_matched:
    # Load the image you want to match
    image_to_be_matched = face_recognition.load_image_file('WhoAreThese/' + check4image)
    print('Looking for a match to this image:', check4image)
    
    # Encode the loaded image into a feature vector
    image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]

    # Iterate over each known image
    for known_image in known_images:
        # Load the known image
        current_image = face_recognition.load_image_file('knownImages/' + known_image)
        
        # Encode the loaded image into a feature vector
        current_image_encoded = face_recognition.face_encodings(current_image)[0]
        
        # Do they match?
        result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)
        print('Checking image with this known image:', known_image)

        if result[0] == True:
            print('Matched:', known_image)
            img_matched = cv2.imread('WhoAreThese/' + check4image)
            img_known = cv2.imread('knownImages/' + known_image)

            cv2.imshow('Matched Image (WhoAreThese)', img_matched)
            cv2.imshow('Matched Image (knownImages)', img_known)
            cv2.waitKey(0)

            break  # If matched, no need to check further for this image
        else:
            print('Not matched:', known_image)
