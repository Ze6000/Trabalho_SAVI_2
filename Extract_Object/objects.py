import cv2
import webcolors 
from ast import literal_eval
import numpy as np


COLOR_HEX_TO_NAMES = {
    '#000000': 'Black',
    '#FFFFFF': 'White',
    '#FF0000': 'Red',
    '#00FF00': 'Lime',
    '#0000FF': 'Blue',
    '#FFFF00': 'Yellow',
    '#FFA500': 'Orange',
    '#800080': 'Purple',
    '#008080': 'Teal',
    '#008000': 'Green',
    '#FFC0CB': 'Pink',
    '#808000': 'Olive',
    '#00FFFF': 'Cyan',
    '#800000': 'Maroon',
    '#FF00FF': 'Magenta',
    '#000080': 'Navy',
    '#808080': 'Gray',
    '#FFD700': 'Gold',
    '#A52A2A': 'Brown',
    '#00FF7F': 'SpringGreen'
}
class Object():
    
    def __init__(self,real_w_x,real_w_y,real_h):
        self.real_w_x = real_w_x
        self.real_w_y = real_w_y
        self.real_h = real_h

    def image(self,point_cloud,hight,width,center):
        self.point_cloud = point_cloud
        self.hight = hight
        self.width = width
        self.center = center
        self.top = int(self.center[1]-(self.hight/2))
        self.bottom = int(self.center[1]+(self.hight/2))
        self.left = int(self.center[0]-(self.width/2))
        self.right = int(self.center[0]+(self.width/2))
        
    def draw_bb(self,image,color):  
        self.color = color
        top_left = (self.top,self.left)
        bottom_right = (self.bottom,self.right+5)
        cv2.rectangle(image, top_left, bottom_right, self.color, 3)

    def lableling(self,lable,image):
        self.lable = lable
        cv2.putText(image,str(lable),(self.top-15,self.right+35),cv2.FONT_HERSHEY_SIMPLEX,0.7, self.color, 2, cv2.LINE_AA)

    def getColor(self,image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the object region (you may need to segment the object beforehand)
        object_region = hsv_image[int(self.hight/4):self.hight-int(self.hight/4),int(self.width/4):self.width-int(self.width/4)]  # Example region of interest


        # Calculate the color histogram of the object region
        histogram = cv2.calcHist([object_region], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # Find the peak value in the histogram
        peak_value = np.unravel_index(histogram.argmax(), histogram.shape)

        # Convert the peak value to HSV color space
        hue = peak_value[0]
        saturation = peak_value[1]

        # Calculate the median brightness value in the object region
        value = 1.5*np.median(object_region[:, :, 2])  # Use the V channel for brightness

        # Convert HSV to RGB
        hsv_color = np.uint8([[[hue, saturation, value]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        rgb_color = rgb_color[::-1]

        rgb_color = literal_eval(str(tuple(rgb_color)))


        try:
            closest_color_name = actual_color_name = webcolors.rgb_to_name(rgb_color)
        except ValueError:
            min_colors = {}
            for key, name in COLOR_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - rgb_color[0]) ** 2
                gd = (g_c - rgb_color[1]) ** 2
                bd = (b_c - rgb_color[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            closest_color_name = min_colors[min(min_colors.keys())]
            actual_color_name = None

        if actual_color_name is None:
            self.color_name = str(closest_color_name)
        else:
            self.color_name = str(actual_color_name)
    

    