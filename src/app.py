import os
import uuid
from flask import Flask, request, render_template, send_from_directory

import uuid
from flask import Flask, send_file
from flask_restful import Api, Resource
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import requests



app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath("mypy.ini"))


@app.route("/")
def index():
      image_names = os.listdir('Images') 
      return render_template("gallery.html", image_names=image_names)


@app.route("/upload", methods=["POST"])
def upload():
    for upload in request.files.getlist("file"):
      
        filename = str(uuid.uuid1()) + os.path.splitext(upload.filename)[1] 

        destination = os.path.join("Images/", filename)
        upload.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)
   # return render_template("complete.html", image_name=filename)
    image_names = os.listdir('Images') 
    return render_template("gallery.html", image_names=image_names)


@app.route('/upload/<filename>')
def send_image(filename): 
    dest = os.path.join(APP_ROOT, 'Images')
    return send_from_directory(os.path.join(APP_ROOT, 'Images'),filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('Images')  
    return render_template("gallery.html", image_names=image_names)

@app.route('/Detector/<imagePath>')
def DetectText(imagePath):
       
        cNN = cv2.dnn.readNet("frozen_east_text_detection.pb")

        def detector(imageR):
            originalImg = imageR
            (H, W) = imageR.shape[:2]

            (newWidth, newHeight) = (640, 320)
            rW = W / float(newWidth)
            rH = H / float(newHeight)

            imageR = cv2.resize(imageR, (newWidth, newHeight))
            (H, W) = imageR.shape[:2]

            layerNames = [ #How the neural network trained
                "feature_fusion/Conv_7/Sigmoid", # text detection confidence
                "feature_fusion/concat_3"] #how the text oriented 

            #check the affects on the image like brightness ...
            #identify the mean of the image
            #set the RBG values
            kMean = cv2.dnn.blobFromImage(imageR, 1.5, (W, H),
                                         (123.68, 116.78,103.94 ), swapRB=True, crop=False)

            cNN.setInput(kMean)

            (scores, geometry) = cNN.forward(layerNames) #extract the scores and geometry from the CNN


            (rowsCount, colsCount) = scores.shape[2:4]
            rectangles = []
            confidences = []

            for y in range(0, rowsCount):

                scoresSet = scores[0, 0, y]
                geometryD0 = geometry[0, 0, y]
                geometryD1 = geometry[0, 1, y]
                geometryD2 = geometry[0, 2, y]
                geometryD3 = geometry[0, 3, y]
                anglesSet = geometry[0, 4, y]

                # loop over the number of columns
                for x in range(0, colsCount):
                    # if our score does not have sufficient probability, ignore it
                    if scoresSet[x] < 0.6:
                        continue

                    # compute the offset factor as our resulting feature maps will
                    # be 4x smaller than the input image
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)

                    # extract the rotation angle for the prediction and then
                    # compute the sin and cosine
                    angle = anglesSet[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    # use the geometry volume to derive the width and height of
                    # the bounding box
                    h = geometryD0[x] + geometryD2[x]
                    w = geometryD1[x] + geometryD3[x]

                    # compute both the starting and ending (x, y)-coordinates for
                    # the text prediction bounding box
                    endX = int(offsetX + (cos * geometryD1[x]) + (sin * geometryD2[x]))
                    endY = int(offsetY - (sin * geometryD1[x]) + (cos * geometryD2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    # add the bounding box coordinates and probability score to
                    # our respective lists
                    rectangles.append((startX, startY, endX, endY))
                    confidences.append(scoresSet[x])

            boxes = non_max_suppression(np.array(rectangles), probs=confidences)

            for (startX, startY, endX, endY) in boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)

                # draw the bounding box on the image
                cv2.rectangle(originalImg, (startX, startY), (endX, endY), (0, 0, 255), 2)
            return originalImg

       

        imageR = cv2.imread(os.path.join(APP_ROOT,"Images/" + imagePath))

        #Convert image to grayscale and Gaussian blur
        #Adaptive threshold
        #Find contours
        #Iterate through contours and filter using contour approximation and area
        #4 means 4 points

        imageT = cv2.imread(os.path.join(APP_ROOT,"Images/" + imagePath))
        gray = cv2.cvtColor(imageT, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,12)
        #smaller regions calculation
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        
        for c in cnts:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            approx2 = cv2.approxPolyDP(c, 0.06 * peri, True)
            if len(approx) == 4  or len(approx2) == 4:
                if area > 3200 :
                    x0,y0,w0,h0 = cv2.boundingRect(approx)
                    cv2.rectangle(imageT, (x0, y0), (x0+w0, y0+h0), (0, 0, 255), 2)
                    #ROI = imageT[y0:y0+h0, x0:x0+w0]
              


          




        originalImg = detector(imageT)
        extens = ".jpg"
        nameF = str(uuid.uuid1())
        fileName = os.path.join(APP_ROOT, 'ImagesResult/' + nameF + extens) 

        cv2.imwrite(fileName, originalImg)

        cv2.destroyAllWindows()

        return send_file(fileName,
                         mimetype="image/jpeg",
                         attachment_filename=nameF + extens,
                         as_attachment=True)



if __name__ == "__main__":
    app.run()
