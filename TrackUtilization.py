import numpy as np
from PIL import Image, ImageDraw
import sys
import os
sys.path.append("../")
import darknet as dn
import math

def detect(imagePath, net, meta):
    # dn.set_gpu(0)
    # net = dn.load_net(b"../cfg/yolov3.cfg", b"../yolov3.weights", 0)
    # meta = dn.load_meta(b"../cfg/coco.data")

    detectname = bytes(imagePath, "utf-8")
    r = dn.detect(net, meta, detectname)

    return r.splitlines()

def createArray(filePath):
    readFile = open(filePath, "r")
    lineList = readFile.readlines()
    arr = []
    for line in lineList:
        words = line.split()
        if words[0] == "2" or words[0] == "7":
            arr.append([float(words[1]), float(words[2])])
    readFile.close()

    arr = np.asarray(arr)

    return arr

def determineUtilization(parkingLocations, detections, imagePath, saveImagePath):
    spotFull = [False] * len(parkingLocations)  #list of parking locations full or empty
    assignment = [-1] * len(detections) #list to match the detections in the image and the index of the parking spot to which they belong
    distance = [0] * len(detections)    #list of the distance between the detection and the parking spot to which it is assigned
    
    #loop through parking spots and determine which detection/car is closest to spot
    for spot in range(0, len(parkingLocations)):
        spotX = parkingLocations[spot][0]
        spotY = parkingLocations[spot][1]

        minDistance = sys.maxsize
        
        for detectionPoint in range(0, len(detections)):
            #calculate distance between car detection and parking spot 
            currDistance = math.sqrt(((spotX - detections[detectionPoint][0])**2) + ((spotY-detections[detectionPoint][1])**2))
            #find closest car to spot 
            if(currDistance < minDistance):
                minDistance = currDistance
                #currpoint is index of car closest to parking spot
                currPoint = detectionPoint

        #check if car has been assigned to spot
        if len(detections) > 0:
            if assignment[currPoint] == -1  and minDistance < 0.05:
                assignment[currPoint] = spot
                #saving info of how far car is from spot in array
                distance[currPoint] = minDistance
                spotFull[spot] = True
                #if car is already assigned, figure out which spot its assigned to
            elif minDistance < 0.05:
                if(minDistance < distance[currPoint]):
                    distance[currPoint] = minDistance
                    #if closer to new spot, change old spot to empty and new spot to full
                    spotFull[assignment[currPoint]] = False
                    spotFull[spot] = True
                    assignment[currPoint] = spot

    fullSpots = []
    emptySpots = []
    
    #fill full and empty spot lists
    for spot in range(0, len(parkingLocations)):
        if spotFull[spot]:
            fullSpots.append(parkingLocations[spot])
        else:
            emptySpots.append(parkingLocations[spot])


#drawing if full or empty
    img = Image.open(imagePath)
    for point in detections:
        x = point[0] * img.width
        y = point[1] * img.height
        draw = ImageDraw.Draw(img)
        location = [x, y, x+15, y+15]
        draw.ellipse(location, fill = (0, 0, 255))
    for point in emptySpots:
        x = point[0] * img.width
        y = point[1] * img.height
        draw = ImageDraw.Draw(img)
        location = [x, y, x+10, y+10]
        draw.ellipse(location, fill = (255, 0, 0))
    for point in fullSpots:
        x = point[0] * img.width
        y = point[1] * img.height
        draw = ImageDraw.Draw(img)
        location = [x, y, x+10, y+10]
        draw.ellipse(location, fill = (0, 255, 0))
    img.save(saveImagePath)

    # return spotFull
    return len(fullSpots)/(len(fullSpots) + len(emptySpots))

def createArray(detectionList):

    #open detections file
    # readFile = open(filePath, "r")
    #adds each line of file to list as list element 
    # lineList = readFile.readlines()
    #list to hold coordinates
    arr = []
    for line in detectionList:
        if line != []:
        #split() breaks line on space
            words = line.split()
            # print(words)
            # print("\n..........\n")
            if words[0] == "2" or words[0] == "7":
                #if valid argument, add elements to list
                arr.append([float(words[1]), float(words[2])])
    # readFile.close()

    #converting list to numpy array for function requirements
    arr = np.asarray(arr)

    return arr

def main():

    #yolo required 
    dn.set_gpu(0)
    net = dn.load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = dn.load_meta(b"cfg/coco.data")


    numArgs = len(sys.argv)
    if numArgs != 4:
        print("Usage: python3 Utilization.py 'pathToParkingLocations.npy' 'pathToImagesFolder' 'pathToSaveFolder'")
        return
    
    parkingLocationsPath = sys.argv[1]
    imagesFolder = sys.argv[2]
    saveFolder = sys.argv[3]

    parkingLocations = np.load(parkingLocationsPath)

    for image in os.listdir(imagesFolder):
        imageName = image[:-4]
        imagePath = imagesFolder + "/" + image
        saveImagePath = saveFolder + "/" + imageName + "_utilization.jpg"
        detections = detect(imagePath, net, meta)
        detections = createArray(detections)
        determineUtilization(parkingLocations, detections, imagePath, saveImagePath)

    # # image
    # #get path to image you want utilizaton of 
    # imagePath = directoryPath + "/image.jpg"
    # #save marked up image
    # saveImagePath = directoryPath + "/utilization.png"
    # #run yolo on the single image
    # detections = detect(imagePath, net, meta)
    # #saving single detections 
    # detectionPath = directoryPath + "/singleDetection.txt"
    # file = open(detectionPath, "w")
    # file.write(detections)
    # file.close()
    # #create array from single detection file 
    # detections = createArray(detectionPath)

    # print(determineUtilization(parkingLocations, detections, imagePath, saveImagePath))

    # # image2
    # imagePath = directoryPath + "/image2.jpg"
    # saveImagePath = directoryPath + "/utilization2.png"
    # detections = detect(imagePath, net, meta)
    # detectionPath = directoryPath + "/singleDetection2.txt"
    # file = open(detectionPath, "w")
    # file.write(detections)
    # file.close()
    # detections = createArray(detectionPath)
    # print(determineUtilization(parkingLocations, detections, imagePath, saveImagePath))

    

if __name__=="__main__":
    main()