import cv2
from capture import CamCapture
import datetime
import time
import os
import argparse

def get_url(link):
    command = 'yt-dlp -g "' + str(link) + '" > stream-url'
    os.system(command)

    with open("stream-url", "r") as file:
        new_url = file.read()
    
    return new_url.strip()

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('youtubeLink', help='Link to youtube livestream')
    parser.add_argument('saveFolder', help='Path to folder to save images')
    return parser.parse_args()



args = parseArguments()
youtubeLink = args.youtubeLink
saveFolder = args.saveFolder

url = get_url(youtubeLink)
url_end_time = time.time() + 60 * 60 * 5 #url only lasts for a few hours

while True:
    if time.time() > url_end_time:
        url = get_url(youtubeLink)
        url_end_time = time.time() + 60 * 60 * 5
    cap = CamCapture(url.strip())
    cap_time, img = cap.read()
    picDate = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    filename = saveFolder + "/" + picDate + ".jpg"
    cv2.imwrite(filename, img)
    cap.close()
    time.sleep(600)