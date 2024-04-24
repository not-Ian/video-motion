import cv2
import os
import numpy as np
from PIL import Image

target_size = (1080, 1080)
stockFootageDirectory = '../../DeepLearning\old deep learning files\DeepLearning2020\datasets\stock_footage/'

# for imagePath in os.listdir(stockFootageDirectory)[0]:
#     print(imagePath)

testVideo = os.listdir(stockFootageDirectory)[4]
# testVideo = "a-baseball-game-in-a-stadium-2430839.mp4"
# testVideo = "a-busy-street-on-a-sunny-day-1625973.mp4"

cap = cv2.VideoCapture(stockFootageDirectory + testVideo)

previousFrame = None
frameRate = cap.get(cv2.CAP_PROP_FPS)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(testVideo)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), frameRate, (frameWidth, frameHeight))

frameIndex = 0
frameOffset = 3
previousFrames = []
addedFrames = []
# previousFrame = None
ret = True

while(ret):
    ret, currentFrame = cap.read()
        
    if (currentFrame is not None):
        modifiedFrame = np.divide(currentFrame, 2)
        previousFrames.append(modifiedFrame)

        if (frameIndex >= frameOffset):
            previousFrame = previousFrames.pop(0)
            previousFrame = -previousFrame + 127
            # np.divide(previousFrame, 2, dtype=np.float64)
            addedFrame = np.add(modifiedFrame, previousFrame).astype(np.uint8)
                        
            addedFrames.append(addedFrame)
            
            if (len(addedFrames) == 2):
                previousFrame = np.divide(addedFrames.pop(0), 2)
                previousFrame = -previousFrame + 127

                modifiedFrame = np.divide(addedFrames[0], 2)
                addedFrame = np.add(modifiedFrame, previousFrame).astype(np.uint8)
            
                out.write(addedFrame)
            else:
                out.write(addedFrame)
        else:       
            out.write(currentFrame)
        
    print(int(frameIndex / numberOfFrames * 100) / 100)
    frameIndex += 1
    
cap.release()
out.release()