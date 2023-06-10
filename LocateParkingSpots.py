import numpy as np
from sklearn.cluster import AgglomerativeClustering
from PIL import Image, ImageDraw

import os
import sys

sys.path.append("../")
import darknet as dn


#this function runs yolo
def detectFolder(imagesFolder):
    #yoloy given functions
    sys.path.append(os.path.join(os.getcwd(),'python/'))

    dn.set_gpu(0)
    net = dn.load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = dn.load_meta(b"cfg/coco.data")

    #open file that we are saving detections to
    # file = open(labelsPath, "w")
    #loops through folder and returns names of images
    files = os.listdir(imagesFolder)

    detections = []

    #loops through every image in 'files'
    for pictureName in files:
        #path to each individual picture
        detectname = imagesFolder + "/" + pictureName
        detectname = bytes(detectname, "utf-8")

        #run detection on each picture
        r = dn.detect(net, meta, detectname)

        #writes class # to file X, Y coordinates and the width, heigth of box 
        detections = detections + r.splitlines()

    return detections




def RunAgglomerative(distance_threshold, array, image):

    #pass arguments
    ag = AgglomerativeClustering(distance_threshold=distance_threshold, compute_full_tree=True, n_clusters = None)
    #call clustering function
    ag.fit(array)
    #puts cluster labels into a list
    labels = ag.labels_

    #set means no duplicates 
    #find how many different groups there are 
    num_clusters = len(set(labels))
    #take away a parking spot if invalid group
    if(-1 in labels):
        num_clusters -= 1

    #new list that is same length as number of clusters
    points_by_cluster = [None]*num_clusters
    #to find the center point of each cluster
    for point in range(len(labels)):
        #assigning indivual points to cluster number
        cluster_number = labels[point]
        #putting our detections from yoloy into car info one at a time
        car_info = [array[point][0], array[point][1]]
        #declaring index for start of list of labels  
        if(points_by_cluster[cluster_number] == None):
            points_by_cluster[cluster_number] = []
            #appedning label to correct list 
        points_by_cluster[cluster_number].append(car_info)

    #now taking lists of labels to calculate center of clusters 
    large_clusters = []
    small_clusters = []

    #loop through each cluster/grouping to calculate center
    for list in points_by_cluster:

        #how many detections of a car were found in that one spot
        num_cars = len(list)

        total_sum_x = 0
        total_sum_y = 0

        #calculating center
        for point in list:
            total_sum_x += point[0]
            total_sum_y += point[1]
        
        center_point = [(total_sum_x/num_cars), (total_sum_y/num_cars)]

        # x_deviation_sum = 0
        # y_deviation_sum = 0
        # for point in list:
        #     x_deviation_sum += (center_point[0] - point[0])**2
        #     y_deviation_sum += (center_point[1] - point[1]) **2
        

        #need at least 100 detections to consider as a parking spot 
        if(num_cars > 100):
            large_clusters.append(center_point)
        else:
            small_clusters.append(center_point)

    if image != "None":
        img = Image.open(image)
        for point in array:
            x = point[0] * img.width
            y = point[1] * img.height
            draw = ImageDraw.Draw(img)
            #draws circle with elipse 
            location = [x, y, x+5, y+5]
            draw.ellipse(location, fill = (0, 0, 255))
        for point in large_clusters:
            x = point[0] * img.width
            y = point[1] * img.height
            draw = ImageDraw.Draw(img)
            location = [x, y, x+5, y+5]
            draw.ellipse(location, fill = (0, 255, 0))
        imageSavePath = image[:-4] + "_clustering.png"
        img.save(imageSavePath)

    return large_clusters

 



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

    numArgs = len(sys.argv)
    if numArgs < 3:
        print("Usage: python3 LocateParkingSpots.py 'pathToImagesFolder' 'pathToSaveFolder'")
        return
    
    imagesFolder = sys.argv[1]
    saveFolder = sys.argv[2]

    imagePath = "None"
    if numArgs == 4:
        imagePath = sys.argv[3]

    #run all images through yolo
    print("Running Detections")
    detections = detectFolder(imagesFolder)
    
    #reads file Created^ and creates array of coordinates for algorithm 
    array = createArray(detections)

    #call clusteing algorithm
    print("Running Algorithms")  
    #holds number of large clusters returned from algorithm   
    results = RunAgglomerative(0.3, array, imagePath)

    #creates and saves numpy array file to use later 
    parkingLocationsPath = saveFolder + "/parkingLocations.npy"    
    parkingLocations = np.asarray(results)
    np.save(parkingLocationsPath, parkingLocations)


if __name__=="__main__":
    main()
