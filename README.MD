# Implementation of Image Inpainting using 
## "Region Filling and Object Removal by Exemplar-Based Image Inpainting " by A. Criminisi, P. pareze and K. Toyama

 1. Mask Creation- Creation of mask on image using mouse cursor as desired by user to mask the areas in image that the user wants to remove.
 2. Inpainting - Inpainting using the above Algorithm with the help of mask.

## Files - 
1. marker.py - This code helps user to select the region of interest on the image i.e. the part user wants to remove.The code creates a new image consisting of black and white region with the removed area as white and remaining image as black to help guide the algorithm.The mask is saved as masked.jpg
2. inpainter.py - is used to inpaint the same image using mask as a reference. The user needs to mention the file path of image. Here user can also see the progress and the time taken to fill a patch
3. system_design_project.py - it is a gui interface which does not show progress but is easy to use. This adds functionality of the above two codes and wrap the operation aroung a GUI.It asks the file location or the image location and then asks the user to draw the mask. The mask is automatically saved and the inpainting begins with default options, the user however can change these values such as patch size etc.The output image in generated in current working directory
