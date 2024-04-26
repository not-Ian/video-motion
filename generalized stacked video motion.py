import cv2
import os
import numpy as np
from PIL import Image

target_size = (1080, 1080)
stockFootageDirectory = '../../DeepLearning\old deep learning files\DeepLearning2020\datasets\stock_footage/'

testVideo = os.listdir(stockFootageDirectory)[4]
# testVideo = "a-baseball-game-in-a-stadium-2430839.mp4"
# testVideo = "a-busy-street-on-a-sunny-day-1625973.mp4"

cap = cv2.VideoCapture(stockFootageDirectory + testVideo)

previousFrame = None
frameRate = cap.get(cv2.CAP_PROP_FPS)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(testVideo, numberOfFrames)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), frameRate, (frameWidth, frameHeight))


def mergeFrames(out, currentLayerSchema, currentFrame):
    currentLayerSchema[1].append(currentFrame)
    
    if (len(currentLayerSchema[1]) > currentLayerSchema[0]):
        previousFrame = np.divide(currentLayerSchema[1].pop(0), 2)
        previousFrame = -previousFrame + 127
        
        modifiedFrame = np.divide(currentFrame, 2)
        addedFrame = np.add(modifiedFrame, previousFrame).astype(np.uint8)
                
        if (len(currentLayerSchema) == 3):
            currentLayerSchema = currentLayerSchema[2]
            mergeFrames(out, currentLayerSchema, addedFrame)
        else:
            out.write(addedFrame)
    else:       
        out.write(currentFrame)

    
frameIndex = 0
currentLayerSchema = (1, [], (1, [], (1, [])))
ret = True

while(ret):
    ret, currentFrame = cap.read()
        
    if (currentFrame is not None):
        mergeFrames(out, currentLayerSchema, currentFrame)
        
    print(int(frameIndex / numberOfFrames * 100), "%")
    frameIndex += 1
    
cap.release()
out.release()