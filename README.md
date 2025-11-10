# Measure Application Backend

## Solution 

### Introduction
This is a django-python based project that allows end user to upload  PET assessment videos store them and process them to get analytics of said videos.
There are multiple assessment test like broad jump, endurace run, sit and throw etc etc, for this project we have divided them into three categories which are as follows
1. Time (where measured quantity is time ex, speed test)
2. Distance (where measured quantity is disntace ex, broad jump)
3. Counter (where measured quantity is counts ex, hexagon test)

As of now we are only able to figure out solution to first two categories, i.e Time and Distance. Time is rather simple which just include adding a timer in application
Distance measurement from video is done via image processing algorithm and AI (Pose estimation)

⚠️ COMMON-PREPROCESSING ⚠️
- ALL IMAGES AND VIDEOS ARE IN OR CONVERTED TO 1280X720 SIZE.
- COLOR BALANCE
- CONTRAST ENHANCEMENT

### Calibration

To estimate distane in image we need to map points in image to real world. We use concept of [homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html), which is
backbone of this application. In short to map image to real world we place 6 points with knowin relative distance on ground and take a image, in that image using image processing we 
detect those points and get their **image** coordiantes in pixel. Since we know their relative distances we can have a real life coordinate corresponding to each pixel cooridinate in image
that allows us to create a Homography MAP. Which is a matrix(2nd order tensor)  allowing us transform vectors in image to real world.


The points are placed on black **calibration sheet** to enhance contrast. algorithm works in three steps show below

<img width="500" alt="image" src="https://github.com/user-attachments/assets/8ac07624-7ba4-40ae-a056-7ffb52e4da32" />


<img width="500" alt="image" src="https://github.com/user-attachments/assets/a17c4ddb-3b72-4ea3-acdf-21e1160577a2" />


<img width="500" alt="image" src="https://github.com/user-attachments/assets/f065c4a7-a024-4309-acd7-dcb43757bd77" />

Example of floor disntaces estimation is shown below , all distances can we cross-checked from real life badminton court distances

<img width="500"  alt="image" src="https://github.com/user-attachments/assets/8d1258ac-5212-408a-b233-d49e24525953" />


### Estimation of required distance

After calibration we can get disntace between given **two points** on floor but how do we get those two points, in case of broad jump we track a marker placed at ankles of player
the starting frame and ending frame is chosen, we also use AI - Pose estimatoin (YOLO) to narrow down on area where we have to search for marker. It is color based object detection
that quantize image into 7 colors and depending on color of marker we chose one of the 7 color to detect marker


https://github.com/user-attachments/assets/8ca7a34d-6a3a-4435-b916-86c3047dce4e

and distance is captured based on positon of marker attached to participants shoes

<img width="500" alt="image" src="https://github.com/user-attachments/assets/00745e56-6119-4234-8ede-4bf8ebd0a698" />


## END POINTS

### /upload_video
Takes a video file input, string input of student name and test type (broad jump, sit and reach etc)

### /list_videos
List all the videos

### /calibrate
Takes a 1 second short video as input and saves homopgaphy matrix if the calibratoin sheet is found in every frame of video

### /calibration_info
Shows detected points

### /get_processed_video
Gets processed video from video id

### /process_image
Takes a image and image coordinate as input, savse the color of pixel at the coordiante, this function is not used in general unless there is color change in markers
of calibration sheet



