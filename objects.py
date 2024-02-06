# Object class
# May ccontain:
#  - Object Point Cloud
#  - Hight 
#  - Width
#  - Other properties 

import cv2

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
        image[self.center] = (0,255,0)
        top_left = (self.top,self.left)
        bottom_right = (self.bottom,self.right+5)
        cv2.rectangle(image, top_left, bottom_right, self.color, 3)

    def lableling(self,lable,image):
        self.lable = lable
        cv2.putText(image,str(lable),(self.top-15,self.right+35),cv2.FONT_HERSHEY_SIMPLEX,0.7, self.color, 2, cv2.LINE_AA)

    




    

    