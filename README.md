## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Screenshot2018-09-1309.53.png "Undistorted"
[image2]: ./Screenshot2018-09-1310.11.png "Road Transformed"
[image3]: ./Screenshot2018-09-135.38.png "Binary Example"
[image4]: ./Screenshot2018-09-1316.png "Warp Example"
[image41]: ./Screenshot2018-09-1316.51.png "Binary warped"
[image5]: ./Screenshot2018-09-1317.49.png "Fit Visual"
[image51]: ./Screenshot2018-09-1411.12.png "Fit Visual Final"
[image6]: ./Screenshot2018-09-1421.35.53.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the `P4_rev.ipynb` in cells 2 through 4. 
For this section first I tried to prepare object points (`objpoints`). This point are going to use as $(x,y,z)$ coordinates of each chessboard corners in the global plane, which in this project it was assumed that each chessboard is fixed on the $(x,y)$ plane at $z=0$. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Codes for this part can be found on cells 6 and 7 of `P4_rev.ipynb`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell 9 in `P4_rev.ipynb`). For this task, Sobel operator (cell 8 `abs_sobel_thresh`) and direction of gradients (cell 8 `dir_threshold`). Next, color thresholding between red and green channel was done (cell 9 line 9 through 14). Then, in line 10 through 18 `S` and `L` channels were thresholded.

Then the results of different thresholding filters were combined. An area of interest was also defined (lane 18)The results of all steps were combined on line      

Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the the source and destination points manually and following table shows my chosen points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

![alt text][image4]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

The function `warper` was defined in cell 12 and here is a example of the results:

![alt text][image41]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In cell 15 the function `laneFinder` is defined. The sliding window based histogram approach was used on bottom half of the images and found the lane locations in a 10 layers subsegmentation of the image (Below image).
![alt text][image5]

Then the `polyfit` functions was used to fit the polynomials to to both right and left sides of each image. the output looked like this on the test images:

![][image51]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature and the distance from the lane center was calculated in cell 19 (`curveFinder` function) and as described in lectures.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cells 20 through 21 in my code in `P4_rev.ipynb` in the function `outputMaker()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output3.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems I encountered were due to lighting conditions and shadows. TO solve these issues thresholds were defined by using both RGB and HLS channels. By combining these thresholds I was able to avoid shadowing (cell 9). The other main issues that I encountered was with the unrealistic values for the curvature. This issue was also solved by changing the destination points for warping the images (cell 12). 
