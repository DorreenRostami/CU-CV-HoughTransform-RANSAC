# CU-CV-HoughTransform-RANSAC
Part 1) <br/>
I used the canny edge detector in skimage. The default sigma value (which is 1) has the output similar to the one in the handout. <br/>
Also to get more familiar with Hough transformation and how it is implemented, I used this website: https://alyssaq.github.io/2014/understanding-hough-transform/<br/>
For zeroing the values in the accumulator around the blue line, the neighborhood radius was found through trial and error. At value 500 it found the lane, but it was a line a little bit to the left of the line shown in the image in the handout. Value 530 gave the same output as the image in the handout. <br/>
![image](https://github.com/user-attachments/assets/70e7f3a0-6aba-4466-b9da-63c00c8b75a6) <br/>
<br/>
Part 2) <br/>
I implemented matchPics with the help of these:<br/>
https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/ <br/>
https://stackoverflow.com/questions/54203873/how-to-get-the-positions-of-the-matched-points-with-brute-force-matching-sift/54220651#54220651 <br/>
And this is the output (the 170 matches with the least distances were chosen so the output looks similar to the one in the handout. For example, there would’ve been a line connecting the book cover to some lines in the background if I used 180 as the threshold) <br/>
![image](https://github.com/user-attachments/assets/716d3098-4631-4202-898d-1ad7ffbb69c5) <br/>
I used https://math.stackexchange.com/a/3511513 for transforming a point using the homography matrix. <br/>
Sometimes the outputs aren’t that good since RANSAC randomly chooses the points, but the outputs shown here are two of the good ones that appear more often:<br/>
matching result after RANSAC: <br/>
![image](https://github.com/user-attachments/assets/7fac52af-391d-40b5-8260-01fcb017dd50)<br/>
visualization of bounding box:<br/>
![image](https://github.com/user-attachments/assets/441d92b1-3e35-4b48-a7f8-5b58d020aed1)<br/>
final result: <br/>
![image](https://github.com/user-attachments/assets/ca57c454-a707-438e-a651-c8eb9d15cbc4)<br/>
H matrix: <br/>
[[ 2.43004075e-03 -1.15499041e-03  7.76392521e-01]<br/>
 [ 1.23074754e-05  6.78541589e-04  6.30235170e-01]<br/>
 [ 8.15056025e-08 -3.09772918e-06  3.25329662e-03]]<br/>
============================================<br/>
Another output:<br/>
matching result after RANSAC:<br/>
![image](https://github.com/user-attachments/assets/a4299711-9ae6-451a-98a4-d06209d5b383)<br/>
visualization of bounding box:<br/>
![image](https://github.com/user-attachments/assets/1d5f69ab-8851-4292-bdf7-049285c25699)<br/>
final result: <br/>
![image](https://github.com/user-attachments/assets/fa7960db-7a51-4ba9-934c-1118292422d2) <br/>
H matrix:<br/>
[[ 2.25360248e-03 -1.14048451e-03  7.73124991e-01]<br/>
 [-8.38370008e-05  6.24569429e-04  6.34240283e-01]<br/>
 [-1.60783866e-07 -3.04665629e-06  3.19925362e-03]]<br/>
