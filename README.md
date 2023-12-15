This project involves an image processing and character recognition program designed to detect and recognize vehicle license plates. Below are detailed pieces of information about the project.

## **Key Points**
#### 1. Image Preprocessing
Convert the image to grayscale using cv2.cvtColor.
Apply Gaussian blur and adaptive thresholding using cv2.GaussianBlur and cv2.adaptiveThreshold for effective image preprocessing.

#### 2. Contour Detection:
Find contours in the image using cv2.findContours.
Visualize the detected contours using cv2.drawContours.

#### 3. License Plate Candidate Selection:
Choose contours that are likely to represent license plates based on criteria such as area, width, height, and aspect ratio.

#### 4. License Plate Region Rotation and Extraction:
Rotate and crop the selected license plate region to extract an accurate license plate image.

#### 5. Character Recognition using Tesseract:
Utilize pytesseract.image_to_string to perform character recognition on the license plate image.
#### 6. Clustering and Visualization:
Use the find_chars function for contour clustering to identify valid license plate regions.
Visualize the clustering results to inspect the final outcome.

 ## **Packages Used and Their Versions:**
 1. python          (3.X)
 2. opencv          (4.X)
 3. numpy
 4. matplotlib
 5. pytesseract


### References
https://github.com/kairess/license_plate_recognition