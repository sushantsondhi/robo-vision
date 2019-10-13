Robotics Technology Course (JRL302) Assignment#1 on Robotics Vision

## Camera Calibration and Structure Computation

The assignment can be found at: http://www.cse.iitd.ac.in/~sumantra/courses/cv/assignment_3.html

### Part1: Using a Calibrated Stereo Camera System
----------------------------------------------
1. Given: The camera calibration matrix parameters (i.e. M = A[R|T] & M' = A'[R'|T'])
Task1: Find Fundamental Matrix
Procedure: 
    - Mark corresponding points in two images and get their coordinates (Get 5  such points)
    - Solve p'<sup>T</sup>Fp = 0 for 5 different points and calculate F
    - F = (A'<sup>-1</sup>)<sup>T</sup>EA<sup>-1</sup> (where E = cross(T,R)) gives fundamental matrix using calibration parameters
  	Task2: Draw Epipolar Line
  	Procedure:
    - Find the equation of the line joining optical centre of left camera (C<sub>l</sub>) and any point in the left image
    - Find the image of the line in the right camera using M'
    - This gives the epipolar line for the chosen point
    - Draw the line for 5 different points

2. Given: Part1.1
Task1: Make a wireframe model of two objects on a black background
Procedure:
    - Take a point in the left image and corresponding point in the right image
    - Using the equations P<sub>W</sub> = R(λp) + T = R'(λ'p') + T', find λ
    - Use λ to compute P<sub>W</sub> = R(λp) + T
    - Repeat this for all points on all the boundaries
    - Use these points to construct a wireframe model
    - Use the wireframe model to get Top and Front views

### Part2: Calibrating Cameras
--------------------------
1. Given: Two images and their 2D and 3D coordinates of important points
Task1: Calculate M = A[R|T] & M' = A[R|T']
Procedure:
    - Find corresponding 2D image coordinates of given 100 3D points
    - Calculate M using equation λp = MP<sub>W</sub>
    - Estimate 11 camera parameters from M using 6 step procedure
    - Using the least square method calculate 2D pixel error and 3D object space error.
