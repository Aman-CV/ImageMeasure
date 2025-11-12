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

To estimate distance in image we need to map points in image to real world. We use concept of [homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html), which is
backbone of this application. In short, to map image to real world we place 6 points with know relative distance on ground and take a image, in that image using image processing we 
detect those points and get their **image** coordiantes in pixel. Since we know their relative distances we can have a real life coordinate corresponding to each pixel cooridinate in image
that allows us to create a Homography MAP. Which is a matrix(2nd order tensor) allowing us transform coordinates in image to real world.


The points are placed on black **calibration sheet** to enhance contrast. algorithm works in three steps show below

<img width="500" alt="image" src="https://github.com/user-attachments/assets/8ac07624-7ba4-40ae-a056-7ffb52e4da32" />


<img width="500" alt="image" src="https://github.com/user-attachments/assets/a17c4ddb-3b72-4ea3-acdf-21e1160577a2" />


<img width="500" alt="image" src="https://github.com/user-attachments/assets/f065c4a7-a024-4309-acd7-dcb43757bd77" />

Example of floor distances estimation is shown below , all distances can be cross-checked from real life badminton court distances

<img width="500"  alt="image" src="https://github.com/user-attachments/assets/8d1258ac-5212-408a-b233-d49e24525953" />


### Estimation of required distance

After calibration we can get distance between **two points** on floor but how do we get those two points, in case of broad jump we track a **marker** placed at ankles of participant
the starting frame and ending frame of jump is chosen, we also use AI - Pose estimatoin (YOLO) to narrow down on area where we have to search for marker. It is color based object detection that quantize image into 7 colors and depending on color of marker we chose one of the 7 color to detect marker.


https://github.com/user-attachments/assets/8ca7a34d-6a3a-4435-b916-86c3047dce4e

and distance is captured based on positon of marker attached to participants shoes

<img width="500" alt="image" src="https://github.com/user-attachments/assets/00745e56-6119-4234-8ede-4bf8ebd0a698" />


## END POINTS

### POST /upload_video
Uploads a participant’s video. Depedning on whether test type requires distance calculation or not, the video is sent for processing. Should the assessment_id, test_id and participant_id already exist in database the function will update that instead of creating new object

Form data
- `video` (File, required) – Video file
- `participant_name` (String, optional) – Default: "NoName"
- `pet_type` (String, optional) – Default: "BT"
- `duration` (Float, optional) – Default: 0
- `assessment_id` (String, optional) - Default : dummy
- `participant_id` (String, optinoal) - Default : dummy
- `to_be_progressed` (bool) - Default : False (if its true the measured quantity is distance)

Success response 
```
{ "status": "success", "name": "video.mp4", "participant_name": "John", "pet_type": "BT" }
```

Error response
```
{ "status": "error description" }
```



### GET /list_videos
Returns list of processed videos

Success Response
```
{
  "videos": [
    {
      "id": 1,
      "name": "test.mp4",
      "file": "/media/videos/test.mp4",
      "participant_name": "John",
      "pet_type": "BT",
      "distance": 12.5,
      "is_processed": true,
      "progress": 100,
      "duration": 8.5,
      "to_be_processed": false
    }
  ]
}
```

Error Response
```
{"status": "error description"}
```


### POST /calibrate
Uploads a calibration video, extracts a frame, detects calibration points, and computes a homography matrix.

Form data
- `video` (File, required) – Calibration video
- `square_size` (Float, optional) – Unit distance (default: 0.984252)

Success Response 
```
{ "status": "success" }
```

Error Response
```
{"status": "error description"}
```
### GET /calibration_info
Returns stored calibration details including matrix and image URLs.

Success Response
```
{
  "status": "success",
  "calibration_info": {
    "square_size": 0.984252,
    "matrix_url": "/media/homography.json",
    "image_url": "/media/calibrated.jpg"
  }
}
```

Error Response
```
{"status" : "error desription"}
```

### GET /get_processed_video
Fetches processed video URL by ID.

Query Parameter:
- `id` (Integer, required) – Video ID

Success Response
```
{
  "status": "success",
  "processed_video_url": "http://example.com/stream_video/output.mp4/"
}
```

Error Response
```
{"status" : "error description"}
```

### GET /list_videos_by_assessment_and_test
Fetch and return list of uploads fileterd by assessment_id and test_id

Query Params:
`assessment_id` (String, required) - default dummy
`test_id` (String, required) - default jump

Success Response
```
{
  "videos": [
    {
      "name": "Jump Trial 1",
      "file": "https://example.com/media/videos/jump1.mp4",
      "distance": 3.2,
      "participant_name": "Buddy",
      "pet_type": "broad jump",
      "id": 12,
      "is_processed": true,
      "progress": 100,
      "duration": 15.4,
      "to_be_processed": false,
      "participant_id": "P123"
    }
  ]
}
```

### POST /process_image
Takes a image and image coordinate as input, save the color of pixel at the coordiante, this function is not used in general unless there is color change in markers of calibration sheet

Form Data:
- `image` (File, required) – Input image file
- `x` (Integer, required) – X-coordinate of the selected point
- `y` (Integer, required) – Y-coordinate of the selected point

Success Response
```
{
  "hsv": { "h": 120, "s": 85, "v": 200 },
  "image_base64": "data:image/jpeg;base64,<encoded_image>"
}
```

Error Response
```
{"status": "error description"}
```





