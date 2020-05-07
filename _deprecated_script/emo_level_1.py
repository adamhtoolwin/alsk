#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math


def emo_lev(arousal,valence):    #Function that returns emotion and intensity
    r = 5
    x, y = valence, arousal
    dist = (x**2+y**2)**0.5

    #Calc theta in radian
    theta = math.atan(y/x)

    #Convert theta from rad to deg
    theta = 180 * theta/math.pi

    def lev(dist):
        if(dist<2.5):
            return "Low"
        #if(dist>1.67 and dist<3.34):
            #return "Medium"
        else:
            return "High"
        
    #Check which intensity
    levl = lev(dist)


    def conv_emo(theta):
        if(x>0 and y>0):  #1st Quadrant
            if theta <= 90:
                return "Joy or Happiness",1,theta  #1st Quad, 1st half
        else:
            if(x>0 and y<0): #4
                theta+=360
                quad = 4
                if theta <= 360:
                    return "Tenderness",4,theta

            else:
                if(x<0 and y>0): #2
                    theta+=180
                    quad = 2
                    if theta <= 180:
                        return "Anger or Fear",2,theta
                else:
                    if(x<0 and y<0): #3
                       theta+=180
                       if theta <= 270:
                            return "Sadness",3,theta
                       
                   
    #print("Theta before conversion: {}".format(theta))
    emo,quad,theta = conv_emo(theta)
    #print("Distance = {} \nTheta after conversion = {} \nquadrant = {} ".format(dist,theta,quad))
    return emo,levl,theta,quad    #END of emo_lev

#Input the valence, arousal values here.
valence_x, arousal_y = -3,-2
emotion, level, theta,quadrant = emo_lev(arousal_y,valence_x)

#Optional printing. Just return the values you need if you want.
print("Emotion    : "+emotion)
print("Intensity  : "+level)
print("Theta      : {} degree".format("%.2f"%theta))
print("Quadrant   : {}".format(quadrant))

