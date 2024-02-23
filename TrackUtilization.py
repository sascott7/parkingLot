import numpy as np
from PIL import Image, ImageDraw
import argparse
import torch

import sys
import os
import math

import utility

def main():
    args = parseArguments()
    
    parkingLocationsPath = args.parkingLocationsPath
    imagePath = args.imagesFolder
    saveFolder = args.saveFolder

    parkingLocations = np.load(parkingLocationsPath)

    if args.d:
        for image in os.listdir(imagePath):
            imageName = image[:-4]
            singleImagePath = os.path.join(imagePath, image)
            saveImagePath = saveFolder + "/" + imageName + "_utilization.jpg"
            determineUtilization(parkingLocations, singleImagePath, saveImagePath)
    else:
        imageName = imagePath[:-4]
        saveImagePath = saveFolder + "/" + imageName + "_utilization.jpg"
        determineUtilization(parkingLocations, imagePath, saveImagePath)

def determineUtilization(parkingLocations, imagePath, saveImagePath):
    model = utility.load_model()
    img = Image.open(imagePath)
    detections = utility.detect(img, model)
    detections = utility.createArray(detections)
    calculateUtilizationRate(parkingLocations, detections, imagePath, saveImagePath)

def calculateUtilizationRate(parkingLocations, detections, imagePath, saveImagePath):
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

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('parkingLocationsPath', help='path to .npy parking lot location file')
    parser.add_argument('imagePath', help='path to image or folder of images to track utilization')
    parser.add_argument('-s', '--save', dest='saveFolder', help='Folder to save information')
    parser.add_argument('-d', '--directory', action='store_true')
    return parser.parse_args()

if __name__=="__main__":
    main()