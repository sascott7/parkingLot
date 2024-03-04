import numpy as np
from PIL import Image, ImageDraw
import argparse
from sklearn.neighbors import KDTree

import os

import utility

def main():
    args = parseArguments()
    
    parkingLocationsPath = args.parkingLocationsPath
    imagesPath = args.imagesPath
    saveFolder = args.saveFolder
    if saveFolder == None:
        saveFolder = imagesPath

    parkingLocations = np.load(parkingLocationsPath)

    for image in os.listdir(imagesPath):
        imageName = image[:-4]
        imagePath = os.path.join(imagesPath, image)
        saveImagePath = saveFolder + "/" + imageName + "_utilization.jpg"
        determineUtilization(parkingLocations, imagePath, saveImagePath)

def determineUtilization(parkingLocations, imagePath, saveImagePath):
    model = utility.load_model()
    img = Image.open(imagePath)
    detections = utility.detect(img, model)
    detections = utility.createArray(detections)
    calculateUtilizationRate(parkingLocations, detections, imagePath, saveImagePath)

def calculateUtilizationRate(parkingLocations, detections, imagePath, saveImagePath):
    spotFull = [False] * len(parkingLocations)

    spotTree = KDTree(parkingLocations)
    detectionsTree = KDTree(detections)

    spotDistance, spotNeighbors = detectionsTree.query(parkingLocations, k=1)
    detectionDistance, detectionsNeighbors = spotTree.query(detections, k=1)

    for i in range(len(parkingLocations)):
        spotNeighbor = spotNeighbors[i]
        if detectionsNeighbors[spotNeighbor] == i:
            if spotDistance[i] < 0.04:
                spotFull[i] = True

    fullSpots = []
    emptySpots = []
    
    #fill full and empty spot lists
    for spot in range(len(parkingLocations)):
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

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('parkingLocationsPath', help='path to .npy parking lot location file')
    parser.add_argument('imagesPath', help='path to image or folder of images to track utilization')
    parser.add_argument('-s', '--save', dest='saveFolder', help='Folder to save information')
    return parser.parse_args()

if __name__=="__main__":
    main()