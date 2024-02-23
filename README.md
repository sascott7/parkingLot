# Parking Lot Utilization
This project tracks the utilization of a parking lot from images. The location of the parking spots are dynamically identified before tracking utilization

## Requirements
Install required python packages:
```console
$ pip install -r requirements.txt
```

### Yolov5

Follow the instructions on Pytorch to install torch with cuda support. https://pytorch.org/get-started/locally/

## Capture Requirements
To run capture_livestream to download images from a YouTube livestream download yt-dlp:
https://github.com/yt-dlp/yt-dlp/wiki/Installation

Install required python packages:
```console
$ pip install -r capture_requirements.txt
```

## Use

### Locate Parking Spots

The locate parking spots script takes a folder of images and saves a numpy array with the identified parking spot locations.

The script can be run with the following:
```
$ python LocateParkingSpots.py path_to_images_directory
```
Optional arguments:

-s path to directory to save files

-i path to image to draw parking spot locations and detections

### Track Utilization

The track utilization script takes the parking spot locations and a folder with images and returns the utilization rate of those images.

The script is run with the following:
```
$  python TrackUtilization.py path_to_saved_np_array_location path_to_images_directory
```

Optional arguments:

-s path to directory to save files

### Capture

The capture script will save images from a youtube livestream every two minutes. 

The script is run with the following:
```
$ python capture_livestream.py youtube_livestream_url path_to_directory_to_save_images
```