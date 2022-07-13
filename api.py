import flask
from flask import send_file

import os
import cv2
import numpy as np
app = flask.Flask(__name__)
app.config["DEBUG"] = True


prototxt_path = os.path.join('./deploy.prototxt')
caffemodel_path = os.path.join('./weights.caffemodel')
face_extractor = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
@app.route('/', methods=['GET'])
def home():
    return "<h1>Ongoing Progress</h1><p>This site is a prototype API for for PKMKC</p>"


@app.route('/extract_faces', methods=['POST'])
def extract_faces():
    # request multiple images 
    output = {
        
    }
    images = flask.request.files.getlist('file')

    for image in images:
        file_name = image.filename
        file_extension = file_name.split('.')[-1]
        file_name = file_name.split('.')[0]

        # read image
        imgstr = image.read()
        npimg = np.fromstring(imgstr, np.uint8)
        raw_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # save raw_image
        cv2.imwrite("./raw_images/" + file_name + "." + file_extension, raw_image)


        (h, w) = raw_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(raw_image, (600, 600)), 1.0, (600, 600), (104.0, 177.0, 123.0))
        face_extractor.setInput(blob)
        detections = face_extractor.forward()

        # loop over the detections
        count = 0
        for i in range(0, detections.shape[2]):
            
            print("image number " + str(count))
            
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = raw_image[startY:endY, startX:endX]

                output[file_name + "_" + str(count) + "." + file_extension] = {
                    "face": face.tolist(),
                    "confidence": str(confidence)
                }

                print(output)
                # cv2.imwrite("./faces/" + file_name + "_" + str(count) + ".jpg", face)
                count += 1

    # return the output
    return output



app.run()