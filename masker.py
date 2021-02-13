import numpy as np
import cv2 
import sys

class marker:
    def __init__(self,window_name, destination, color_function):
        self.prev_point = None
        self.window_name = window_name
        self.destination = destination
        self.color_function = color_function
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.window_name,self.on_mouse)
        
    def show(self):
        cv2.imshow(self.window_name, self.destination[0])
        cv2.imshow(self.window_name+"Mask", self.destination[1])
        
    def on_mouse(self,event, x,y,flags,param):
        initial_point = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_point = initial_point
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_point = None
        
        if self.prev_point and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.destination, self.color_function()):
                cv2.line(dst,self.prev_point,initial_point, color, 29)
                self.dirty = True
                self.prev_point = initial_point
                self.show()
        
    
def main():
    img = cv2.imread("windmil.jpg", cv2.IMREAD_COLOR)
    image_copy = img.copy()
    
    inpaintmask = np.zeros(img.shape[:2],np.uint8)
    
    sketch = marker('Image',[image_copy,inpaintmask],lambda :((255,255,255),255))
    while True:
        ch = cv2.waitKey(0)
     
        if ch == ord('w') or ch == ord('W'):
            mask_new = cv2.cvtColor(inpaintmask,cv2.COLOR_BGR2RGB)
            cv2.imwrite("masked.jpg",mask_new)
            print("mask saved")
            break
        else:
            print("Try to mask again")
            break
if __name__ == '__main__':
    print("Press w to save the mask")
    main()
    cv2.destroyAllWindows()