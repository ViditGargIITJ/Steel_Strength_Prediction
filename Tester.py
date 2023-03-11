import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
finalimg = cv2.imread('D:\Programming\pythonProjects\lightAlloys_ML\images\square.jpg')
# cv2.imshow("Original", finalimg)
cv2.waitKey(0)
df = pd.read_csv("DATAcsv.csv")
df = df[df["minRadius"] > 0]
print(df)
# df = df[df["Radius"] > 0]
r = list(df["Row"])
c = list(df["Column"])
# radius = list(df["Radius"])
# radius = [int(i) for i in radius]
# r = [int(i) for i in r]
# c = [int(i) for i in c]
minRadius = list(df["minRadius"])
# meanRadius = list(df["meanRadius"])
maxRadius = list(df["maxRadius"])
for i in  range(len(df)) :
    cv2.circle(img = finalimg,center = (c[i],r[i]),radius = minRadius[i],color =  (255,0,0), thickness = 1 )
    # cv2.circle(img = finalimg,center = (c[i],r[i]),radius = maxRadius[i],color =  (255,255,255), thickness= 1 )
    # cv2.circle(img = finalimg,center = (c[i],r[i]),radius = meanRadius[i],color =  (100,100,100), thickness=1 )
    # cv2.circle(img = finalimg,center = (c[i],r[i]),radius = radius[i],color =  (255,0,0), thickness=1 )
    cv2.circle(img = finalimg,center = (c[i],r[i]),radius = 0,color =  (0,0,255), thickness= 3 )


plt.hist(df["minRadius"])
plt.show()

cv2.imshow("Original", finalimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.scatter(df['Column'],df['Row'],s = 2,color = 'r')
plt.imshow(finalimg)
plt.show()