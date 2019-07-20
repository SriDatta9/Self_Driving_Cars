                                                      # Finding Lane Lines on the Road

The goals of the project are the following:

* Make a pipeline that finds lane lines on the road utilizing gray_scale, Gaussian_blur, Canny_eddge, Hough_transform, extrapolation.
* Reflect on your work in a written report

# Pipeline description

This is done in 7 steps:
step1 : Convert colorimage to gray_scale image.
input - RGB image 
output - gray_scale image

step2 : Reduce the noise and smoothen image using Gaussian blur(kernel_size).
input - gray_scale image
output - smoothened image

step3 : Use the Canny method to identify edges in the image(using higher and lower threshold)
input - smoothened image
output - edges

step4 : Section of the image is masked using Region of Interest.
input - Canny image
output - masked_image

step5 : The masked section is hough transformed to pick out the continuous lines.
input - masked image
output - gives the continuous lines through which we can get the coordinates of the lines.

step6 : The slopes and intercepts of all the lines are averaged out to single slope and intercept using average_slope_intercept, polyfit and average function.
input - lines
output - slopes between -1 and -0.5 are fitted into left fit and averaged out to a single left lane. Slopes between 0.5 and 0.8 are fitted          into right fit and averaged out to a single right lane.

step7 - Using the make_coordinates, draw_line and weighted_image functions, draw the left and right lines on the original image tracking the left and right lanes.
input - slope and intercept of left and right line.
output - lane lines on the road tracking the lanes.

# Potential Shortcomings

* It doesn't function properly, if the roads are very reflective.
* It loses its precision if the roads have sharp turns as there will be less slopes satisfying the cconstraints.

# Possible improvements

* It should predict the turn well in advance. Fitting the curve with high degree polynomial can help in the sharp turns.
* Smoothing out the jitters and if no segments are detected, it should take the previous one and continue till it found the next line.






