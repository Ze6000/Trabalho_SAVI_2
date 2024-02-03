# Object class
# May ccontain:
#  - Object Point Cloud
#  - Hight 
#  - Width
#  - Other properties 

import cv2

class Object():
    def __init__(self,point_cloud,hight,width,center):
        self.point_cloud = point_cloud
        self.hight = hight
        self.width = width
        self.center = center
        self.top = int(self.center[1]-(self.hight/2))
        self.bottom = int(self.center[1]+(self.hight/2))
        self.left = int(self.center[0]-(self.width/2))
        self.right = int(self.center[0]+(self.width/2))
        
    def draw_bb(self,image):  
        image[self.center] = (0,255,0)
        # cv2.circle(image, (self.center[1],self.center[0]), 20, (0,255,0), 3)
        top_left = (self.top,self.left)
        bottom_right = (self.bottom,self.right)
        cv2.rectangle(image, top_left, bottom_right, (0,255,0), 3)


    

    