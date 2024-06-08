import os
import face_recognition
import cv2
import matplotlib.pyplot as plt
#make a list  of avaiable images
images = os.listdir('KnownImages')


#load the image we want to match(this is the image we we to idenitfy: load your image
check4image = "testimage3.jpg"

image_to_be_matched = face_recognition.load_image_file('KnownImages/' + check4image)
print('Looking for a match to this image: ', check4image)
print()
print('Run through the datasets of images we have identified')
print()

img = cv2.imread('KnownImages/'+check4image)
cv2.imshow(check4image,img)
#encode the loaded image into a feature vector
image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]


#finally have the image to be recognized in a mathematical representation
#now let's attempt to identify this image
#iterate over each known images

for image in images:
    #Load the image
    current_image = face_recognition.load_image_file('KnownImages/' + image)
    #encode the loaded image into a feature vector
    current_image_encoded = face_recognition.face_encodings(current_image)[0]
    #do they match ?
    result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)
    print('Checking image with this image:', image)
    img = cv2.imread('KnownImages/' + image)
    cv2.imshow(image, img)
    cv2.waitKey(0)
    if result [0] == True:
        print('Matched:' + image)
    else:
        print('Not matched:' + image)
        
    
    
    
    



















































