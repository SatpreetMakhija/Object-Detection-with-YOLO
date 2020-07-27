
#Import necessary packages

import numpy as np 
import time
import cv2
import argparse as argp           #command line parsing module
import os


#construct command-line argument parse. (think of it as arguments of a function but inputted through the terminal)

ap = argp.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-y", "--yolo", required = True, help = "base path to YOLO directory")
ap.add_argument("-c", "--confidence", type = float, default = 0.5 , help = 'minimum probability to filter weak detections')
ap.add_argument("-t", "--threshold", type = float, default = 0.3, help = 'threshold when applying non-maxima suppression')
args = vars(ap.parse_args())

#vars creates a dict with key as the argument names and values as the argument values inputted by the user. use print(args) to see the dictionary.



#Notes on Non-Maxima Suppression 
# It is a method of ensurin that we detect each object only once.
# It does so by ignoring the bounding boxes that signifcantly overlap each other.



#It's time to load COLO (data-set) labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"]) 
LABELS = open(labelsPath).read().strip().split("\n")


#Initialise a list of colors to represent each class of label in the data-set.
np.random.seed(42)   #seed helps keeping the random values constant each time
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = 'uint8')


#Derive path to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


#Load the YOLO object detector
print("[INFO] loading from the disk....")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#Note: dnn (deep neural network) is a module of OpenCV.


#Load input image and find its spatial dimensions
image = cv2.imread(args["image"])
H = image.shape[0]
W = image.shape[1]


#determine only the output layers that we need from YOLO
ln = net.getLayerNames()  #list of all the different layers in the NN
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #did not understand this line------------???????


#construct a blob. Definition of blob: A blob is a 4D numpy array object (images, channels, width, height). The image below shows the red channel of the blob. 
# Blob is the input to the YOLO NN. 
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB = True, crop = False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()



#Timing information
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


#initialise list of detected bounding boxes, confidences and class IDs
boxes = []
confidences = []
classIDs = []


#-----???????
for output in layerOutputs:
	#loop over each of the detections
	for detection in output:
		#extract classID and confidence of the current object
		scores = detection[5:] #it means strarting from the 5th element of the vector, since the first 4 have dimns of the box
		classID = np.argmax(scores)
		confidence = scores[classID]

		#filter out weak predictions
		if confidence > args["confidence"]:
			#scale bounding box relative to the size of the image
			box = detection[:4]*np.array([W, H, W, H]) #------??????
			(centerX, centerY, width, height) = box.astype('int')

			#using (x, y) coordinates to calculate top-left corner of the bounding box------------??????????????
			x = int(centerX - (width/2))
			y = int(centerY - (height/2))



			#update list of bounding box coordinates, confidences and classIDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)



#apply non-maxima suppression to remove overlapping bounding boxes with weaker confidence level
indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])


#Time to draw boxes and class-images


#check at least one detected object exists
if len(indices) > 0:
	for i in indices.flatten():
		#extract bounding boxcoordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		#draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y) , (x+w, x+h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i]) #---------------?????????
		cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#time to show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)



















