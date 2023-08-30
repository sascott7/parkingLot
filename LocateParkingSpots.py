import numpy as np
from sklearn.cluster import AgglomerativeClustering
from PIL import Image, ImageDraw
import argparse

import os

from yolo import darknet as dn
import utility

def main():
    args = parseArguments()
    imagesFolder = args.imagesFolder
    saveFolder = args.saveFolder
    imagePath = args.imagePath

    print("Detecting Vehicles...")
    detections = detectFolder(imagesFolder)
    array = utility.createArray(detections)

    print("Locating Parking Spots...")
    results = RunAgglomerative(0.3, array, imagePath)

    print("Saving Parking Spots...")
    parkingLocationsPath = saveFolder + "/parkingLocations.npy"    
    parkingLocations = np.asarray(results)
    np.save(parkingLocationsPath, parkingLocations)

    print("Done")


#this function runs yolo
def detectFolder(imagesFolder):
    dn.set_gpu(0)
    net = dn.load_net(b"yolo/cfg/yolov3.cfg", b"yolo/yolov3.weights", 0)
    meta = dn.load_meta(b"yolo/cfg/coco.data")

    files = os.listdir(imagesFolder)
    detections = []
    for pictureName in files:
        detectname = imagesFolder + "/" + pictureName
        detections = detections + utility.detect(detectname, net, meta)

    return detections

def RunAgglomerative(distance_threshold, array, image):
    ag = AgglomerativeClustering(distance_threshold=distance_threshold, compute_full_tree=True, n_clusters = None)
    ag.fit(array)
    labels = ag.labels_

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

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('imagesFolder', help='Images directory')
    parser.add_argument('-d', '--directory', dest='saveFolder', help='Directory to save information')
    parser.add_argument('-i', '--image', dest='imagePath', help='Image to plot parking spots')
    return parser.parse_args()

if __name__=="__main__":
    main()
