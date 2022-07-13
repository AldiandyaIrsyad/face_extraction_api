
import json
import cv2
import numpy as np

# read output.json
with open('output.json') as f:
    images = json.load(f)

    for image_name, image_data in images.items():
        print(image_name, image_data['confidence'])

        # check type of image_data['face']


        print(image_data['face'])
        # convert image_data['face'] to list
        face_list = image_data['face']

        # convert face_list to numpy image
        face_list = np.array(face_list)

        # save images to faces
        cv2.imwrite('faces/' + image_name, face_list)
        


        

    
