# Implementation of Image Inpainting using "Region Filling and Object Removal by Exemplar-Based Image Inpainting " by A. Criminisi, P. pareze and K. Toyama

 1. Mask Creation- Creation of mask on image using mouse cursor using opencv
 2. Inpainting - Inpainting using the above Algorithm

## Files - 
1. marker.py - is used to draw lines on imge which the user provides in the code and draw lines in 
                realtime using reference mask. The mask is saved as masked.jpg
2. inpainter.py - is used to inpaint the same image using mask as a reference. The user needs to 
                mention the file path of image. Here user can also see the progress and the time taken to
                fill a patch
3. system_design_project.py - it is a gui interface which does not show progress but is easy to use 
                              stores the mask and output image in current working directory
