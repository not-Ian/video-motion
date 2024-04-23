import cv2
import os
import numpy as np
from PIL import Image

target_size = (1080, 1080)
stockFootageDirectory = '../../DeepLearning\old deep learning files\DeepLearning2020\datasets\stock_footage/'

# for imagePath in os.listdir(stockFootageDirectory)[0]:
#     print(imagePath)

testVideo = os.listdir(stockFootageDirectory)[1]
# testVideo = "a-baseball-game-in-a-stadium-2430839.mp4"
# testVideo = "a-busy-street-on-a-sunny-day-1625973.mp4"

cap = cv2.VideoCapture(stockFootageDirectory + testVideo)

previousFrame = None
frameRate = cap.get(cv2.CAP_PROP_FPS)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), frameRate, (frameWidth, frameHeight))

frameIndex = 0
frameOffset = 1
previousFrames = []

while(cap.isOpened()):
    ret, currentFrame = cap.read()
        
    if (currentFrame is not None):
        modifiedFrame = np.divide(currentFrame, 2)
        previousFrames.append(modifiedFrame)

        if (frameIndex >= frameOffset):
            previousFrame = previousFrames.pop(0)
            previousFrame = -previousFrame + 127
            # np.divide(previousFrame, 2, dtype=np.float64)
            addedFrames = np.add(modifiedFrame, previousFrame).astype(np.uint8)
            
            pixelsByChannel = np.sum(addedFrames, axis=-1)
            
            # positivePixelsToHide = pixelsByChannel < (150 * 3) # This threshold is important
            # positiveChannelsToHide = np.tile(np.expand_dims(positivePixelsToHide, 2), 3)
            # negativePixelsToHide = pixelsByChannel < (80 * 3) # This threshold is important
            # negativeChannelsToShow = np.tile(np.expand_dims(negativePixelsToHide, 2), 3)
            # addedFrames[positiveChannelsToHide] = 0
            
            out.write(addedFrames)
            
            # maskedFrame = currentFrame
            # maskedFrame[positiveChannelsToHide] = 0
            # maskedFrame[negativeChannelsToShow] = currentFrame[negativeChannelsToShow]
            # out.write(maskedFrame)
        else:       
            out.write(currentFrame)
    else:
        cap.close()
        
    print(frameIndex)
    frameIndex += 1
    
cap.release()
out.release()
