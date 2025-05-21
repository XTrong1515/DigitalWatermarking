import numpy as np 
import cv2
import matplotlib.pyplot as plt

watermark = np.zeros((64, 64), dtype=np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(watermark, 'COPYRIGHT', (2, 40), font, 0.35, 255, 1)

_, watermark = cv2.threshold(watermark, 128, 255, cv2.THRESH_BINARY)

cv2.imwrite('watermark2.png', watermark)