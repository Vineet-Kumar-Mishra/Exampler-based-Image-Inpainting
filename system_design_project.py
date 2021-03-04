print("Now Started")
#----The necessary libraries---------------------------------------------------------

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msg
from PIL import Image, ImageTk
import numpy as np
import cv2
from scipy.ndimage import filters as fl # for convolution
from skimage import io, color, filters, feature # for I/O , color conversion and laplace




#----defining a Marker Class to draw Lines or remove objects-----------------------------

class marker:
    # the color function decides the color of line on image

    def __init__(self,window_name, destination, color_function):
        self.prev_point = None
        self.window_name = window_name
        self.destination = destination
        self.color_function = color_function
        self.show()
        cv2.setMouseCallback(self.window_name,self.on_mouse)


    def show(self):
        cv2.imshow(self.window_name, self.destination[0])


    def on_mouse(self,event, x,y,flags,param):
        initial_point = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_point = initial_point
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_point = None
        
        if self.prev_point and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.destination, self.color_function()):
                cv2.line(dst,self.prev_point,initial_point, color, 20)
                self.prev_point = initial_point
                self.show()
 
#---- Designing an Inpainter class to inpaint----------------------------

class inpaint_image:
    def __init__(self, image, mask, patch):
        
        # image and mask to be type casted to int
        
        self.image = image.astype('uint8')
        self.mask = mask.astype('uint8')
       
        # size of the patch ...to be odd..
        
        self.patch = patch
        
        # make copy of image and mask and also we require preference
        # we need to initialize the priority parameters
        
        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)
        
        # the confidence is 1 for unmasked and 0 for masked pixels
        # the dataterm initially to be zero for all pixels
        
        self.confidence = (1-self.mask).astype(float)
        self.data = np.zeros(self.image.shape[:2])
        self.priority = None
        
        #the front is the edge of the removed part calculated using Laplacian
        
        self.front = None
    
    
    # We require a front at each iteration to calculate priority
    # laplace will give edge i.e where white and black meets
    # -ve for black and +ve for white is returned in laplace
    # lets us remain on the edge or the front
    
    def front_finder(self):
        self.front = (filters.laplace(self.working_mask)>0).astype('uint8')
        
    # We need to keep the centre of the patch on the required points
    # We assume a patch around the point
    
    def patch_around_point(self, point):
        
        # patch size is odd so half of the size
        
        half = (self.patch-1)//2
        
        height, width = self.working_image.shape[:2]
        
        # The patch around the point is obtained
        
        patch_centralized = [[max(0, point[0]-half), min(point[0]+half, height-1)],
                             [max(0, point[1]-half), min(point[1]+half, width-1)]]
        
        return patch_centralized
    
    
    # We need to put this patch on any source
    
    def patch_on_source(self,source, patch):
        return source[patch[0][0]:patch[0][1]+1,
                     patch[1][0]:patch[1][1]+1]
    
    
    # We need to save updated image each execution
    
   
    def saver(self):

        # we will remove the target region

        image = self.working_image*(color.gray2rgb((1-self.working_mask)))
        
        # fill the target region with white color
        
        white = (self.working_mask-self.front)*255
        RGB_of_white = color.gray2rgb(white)
        image = image + RGB_of_white 
        io.imsave("restored.jpg", image)
        
        
        

    
    # we need to update the priority at each iterations
    
    def update_priority(self):
        
        #the below functions automatically updates the confidence and data terms
        
        self.confidence_updater()
        self.data_updater()
        
        # the front will help only update the terms inside the updated target region
        
        self.priority = self.confidence*self.data*self.front
    
    # for updating priority we require confidence updates
    
    def confidence_updater(self):
        confidence_updates = np.copy(self.confidence)
        required_front = np.argwhere(self.front == 1)
        
        for point in required_front:
            patch = self.patch_around_point(point)
            confidence_updates[point[0],point[1]] = sum(sum(self.patch_on_source(self.confidence,patch)))/self.area_patch(patch)
            
            self.confidence = confidence_updates
    
    # we need area of patch to get confidence
    
    def area_patch(self,patch):
        return(self.patch)**2
    
    
    # for updating priority we require data updates
    
    def data_updater(self):
        
        # we need he normal and gradient matrix using below functions
        
        normal = self.normal_matrix()
        gradient = self.gradient_matrix()
        
        norm_X_grad = normal*gradient
        
        self.data = np.sqrt(norm_X_grad[:,:,0]**2 + norm_X_grad[:,:,1]**2)
    
    # we require normal matrix for data term
    
    def normal_matrix(self):
        # we will convolve with x and y kernels to get normal
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = fl.convolve(self.working_mask.astype(float), x_kernel)
        y_normal = fl.convolve(self.working_mask.astype(float), y_kernel)
        
        # stack matrices around the depth
        
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        
        # norm matrix 
        
        norm = np.sqrt(y_normal**2 + x_normal**2).reshape(height, width, 1).repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        
        return unit_normal
    
    # We require gradient matrix for data term
    
    def gradient_matrix(self):
        height, width = self.working_image.shape[:2]
        
        # we require a grey image to get gradient
        
        grey_image = color.rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None
        
        # Sobel filter kernel to find gradient
        
        x_ker = [[-1,0,1],[-2,0,2],[-1,0,1]]
        y_ker = [[-1,-2,-1],[0,0,0],[1,2,1]]
        
        grad_x = fl.convolve(grey_image,x_ker)
        grad_y = fl.convolve(grey_image,y_ker)
        gradient_val = np.sqrt(grad_x**2 + grad_y**2)
        
        
        
        # initialize a zero matrix to get the max gradient value
        
        max_gradient = np.zeros([height, width, 2])

        required_front = np.argwhere(self.front == 1)
        
        for point in required_front:
            patch = self.patch_around_point(point)
            patch_x_gradient = self.patch_on_source(grad_x, patch)
            patch_y_gradient = self.patch_on_source(grad_y, patch)
            patch_gradient_val = self.patch_on_source(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape )

            max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]

        return max_gradient
    
    
    # we need to find the highest priority point in image
    
    def highest_priority_point(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point
    
    # We require shape of patch in finding source patch
    
    def patch_shape(self,patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])
        
    
    # we require a source patch i.e exemplar
    
    def required_source_patch(self, target_pixel):
        
        target_patch = self.patch_around_point(target_pixel)
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = self.patch_shape(target_patch)

        best_match = None
        best_match_difference = 0
        
        # we need to convert RGB to LAb
        
        lab_image = color.rgb2lab(self.working_image)

        for x in range(height - patch_height + 1):
            for y in range(width - patch_width + 1):
                source_patch = [[x, x + patch_height-1],[y, y + patch_width-1]]
                
                if self.patch_on_source(self.working_mask, source_patch).sum() != 0:
                    continue

                difference = self.squared_path_difference(lab_image,target_patch,source_patch)

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        
        return best_match
    
    # we require a path difference for patch and source
    def squared_path_difference(self, image, target_patch, source_patch):
        
        mask = 1 - self.patch_on_source(self.working_mask, target_patch)
        rgb_mask = color.gray2rgb(mask)
        target_data = self.patch_on_source(image,target_patch) * rgb_mask
        source_data = self.patch_on_source(image,source_patch) * rgb_mask
        
        squared_distance = ((target_data - source_data)**2).sum()
        return squared_distance
    
    
    # We need to update the image after each inpaint interation
    
    def update_image(self, target_pixel, source_patch):
        
        target_patch = self.patch_around_point(target_pixel)
        pixels_positions = np.argwhere(self.patch_on_source(self.working_mask,target_patch) == 1) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self.patch_on_source(self.working_mask, target_patch)
        rgb_mask = color.gray2rgb(mask)
        source_data = self.patch_on_source(self.working_image, source_patch)
        target_data = self.patch_on_source(self.working_image, target_patch)

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        self.copy_to_patch(self.working_image,target_patch,new_data)
        self.copy_to_patch(self.working_mask,target_patch,0)
    
    # We need to copy new data into the patch
    
    def copy_to_patch(self,destination, destination_patch, data):
        destination[destination_patch[0][0]:destination_patch[0][1]+1,
                    destination_patch[1][0]:destination_patch[1][1]+1] = data
    
    
    
    # We need to check whether the task is finish or not
    def task_finished(self):
        remaining_pixels = self.working_mask.sum()
        # Returns True when done
        if remaining_pixels == 0:
            return True
        else:
            return False
    
    
    # we compute new image using the algorithm
    
    def inpainter(self):
        
        # to get the time algo took to finish each step
        finish_check = True
        while finish_check:
            root.update() # updates the GUI
            self.front_finder()
            self.saver()
            self.update_priority()
            
            target_pixel = self.highest_priority_point()
            source_patch = self.required_source_patch(target_pixel)
            self.update_image(target_pixel, source_patch)
            finish_check = not self.task_finished()
        return self.working_image



#----defining a function to get patch size----------

patch_size_data = None
def activate_patch():
    global patch_size_data
    patch_size_data = patch_size.get()
    msg.showinfo("Patch",f"patch size of {patch_size_data} received ! \n Please press Start Inpainter !")

    


#----Defining the paint_mask function to create mask--------

path = None # same path variable is used multiple times

def paint_mask():

    global path, inpaint_progress_label, inpaint_progress, status, patch_entry, patch_size, patch_button

    status.config(bg = "SpringGreen2", text = "STATUS : ACTIVE")
    
    path = filedialog.askopenfilename()

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    height, width  = image.shape[0], image.shape[1]


    image_copy = np.copy(image)
    inpaintmask = np.zeros(image.shape[:2],np.uint8) # create a mask size of the image

    # object of class Marker is created which draws line

    painter = marker('Image',[image_copy,inpaintmask],lambda:((255,255,255),255))

    # used as a stopping function to complete masking process

    while True:
        ch = cv2.waitKey(0)
        if ch == ord('s') or ch == ord('S'): # condition to save the mask
            mask_new = cv2.cvtColor(inpaintmask,cv2.COLOR_BGR2RGB)
            cv2.imwrite("masked.jpg",mask_new)
            cv2.destroyAllWindows()


            # A message of completion and next instruction is provided

            msg.showinfo("MASK SAVED !","Dear User the mask has been saved Successfully !")
            instruction.set("Please provide a patch size (odd number) &\n Press Start Inpainter to start inpainting")
            
            patch_entry = Entry(root, textvariable = patch_size, fg = "black", relief = GROOVE)
            patch_entry.pack()
            patch_button = Button(root, text="Patch Size", bg = "LightSteelBlue4", padx = 10, pady = 10, command = activate_patch, relief = GROOVE, fg = "black" )
            patch_button.pack()
            inpaint_progress_label = Label(root, text = " ", bg = "SlateGray2", font = ('Comic Sans MS', 12, 'italic'),fg = "black", padx = 10, pady = 10)
            inpaint_progress_label.pack()
            inpaint_button = Button(root, text = "Start Inpainter" ,command = start_inpaint, padx = 10, pady = 10,
                         relief = GROOVE, bg = "LightSteelBlue4", fg = "black")
            inpaint_button.pack()
            status.config(text = "STATUS : INACTIVE", bg = "firebrick2")
            
            break
        else:
            mask_new = cv2.cvtColor(inpaintmask,cv2.COLOR_BGR2RGB)

            cv2.imwrite("masked.jpg",mask_new)
            
            cv2.destroyAllWindows()
            msg.showinfo("TRY AGAIN ","Dear User the mask has not been saved !")
            status.config(bg = "firebrick2",text="STATUS : INACTIVE")
            break
    

#----Defining a start_inpaint function------------------------------

def start_inpaint():

    global path, inpaint_progress_label, status, patch_size_data, output_button

    # Set the Status and label of execution

    status.config(bg = "SpringGreen2", text = "STATUS : ACTIVE")
    inpaint_progress_label.config(text = "Please wait....")

    image = io.imread(path)
    mask = io.imread("masked.jpg", as_gray=True)
    threshold = float(mask.min()+mask.max())/2.0

    # process the mask

    mask[mask>threshold] = 1
    mask[mask<threshold] = 0
    output_image = inpaint_image(image, mask, patch_size_data).inpainter()
    inpaint_progress_label.config(text = "Completed !")
    io.imsave("restored.jpg",output_image)
    msg.showinfo("TASK","Task has been completed \n Please check the result in current directory as restored.jpg !")
    status.config(bg = "firebrick2", text = "STATUS : INACTIVE")


#----Designing the GUI------------------------------------------------

# Creating one button to start removing i.e masking
# Creating second button to start inpainting

root = Tk()
root.geometry("500x500")
root.title("INPAINTER")
root.configure(bg = "SlateGray2")
root.iconbitmap(r'C:\Users\mishr\Desktop\Inpaint Project\icon.ico')

# We require a status bar to show status

status = None
status = Label(text = "STATUS : INACTIVE", bg = "firebrick2", pady = 10, relief = RIDGE)
status.pack(fill = X, side = BOTTOM)

# We require one label to show the task for begining the masking

masking_label = None
masking_label = Label(root, text = "Welcome to Image Inpainter app \n Press Start MASKING to draw on image and \n Press S to Save and Stop", 
                      bg = "SlateGray2", font = ('Comic Sans MS', 12, 'italic'), 
                      fg = "black", padx = 10, pady = 10)

masking_label.pack(fill = X, side = TOP)

# We require a button to start masking and will start the function paint_mask

masking_button = None
masking_button = Button(root,text = "Start Masking" ,command = paint_mask, padx = 10, pady = 10,
                         relief = GROOVE, bg = "LightSteelBlue4", fg = "black")
masking_button.pack(side = TOP)


# Create a entry for the size of patc
patch_entry = None
patch_size = IntVar()
patch_size.set(11)
patch_button = None
# We require a label to instruct user to begin inpainting

intruction_label = None 
instruction = None
instruction = StringVar()
instruction.set(" ")
instruction_label = Label(root,textvariable = instruction, bg = "SlateGray2", font = ('Comic Sans MS', 12, 'italic'),fg = "black", padx = 10, pady = 10)
instruction_label.pack()

# We require a button to start inpainting
inpaint_button =None 
inpaint_progress_label = None

root.mainloop()
