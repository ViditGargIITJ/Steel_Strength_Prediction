import cv2
import numpy as np
import pandas as pd
oriimg = cv2.imread('D:\Programming\pythonProjects\lightAlloys_ML\images\square.jpg', 2)
cv2.imshow("Original", oriimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(oriimg, 127, 255, cv2.THRESH_BINARY)
# converting to its binary form
kernel = np.ones((7,7),"uint8")
finalimg = cv2.dilate(bw_img,kernel)
# cv2.imshow("Binary", finalimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
info = [[[1,1,1,1] for i in range(len(finalimg[0]))] for j in range(len(finalimg))]
visited = [[0 for i in range(len(finalimg[0]))] for j in range(len(finalimg))]
for i in range(len(finalimg)):
    for j in range(len(finalimg[0])) :
        if ( finalimg[i][j] == 0):
            finalimg[i][j] = 1
        else :
            finalimg[i][j] = 0

def down(i,j):
    length = 0
    for row in range(i+1,len(finalimg)):
        if(finalimg[row][j] == 1) :
            length += 1
        else :
            return length
    return length 


def right(i,j):
    length = 0
    for column in range(j+1,len(finalimg[0])):
        if(finalimg[i][column] == 1) :
            length += 1
        else :
            return length
    # print(length)
    return length 

def fill(i,j):
    if(j - 1 >= 0 and finalimg[i][j-1] == 1) :
        info[i][j][0] = info[i][j-1][0] + 1
        info[i][j][2] = down(i,j)
    else :
        # print(True)
        info[i][j][2] = down(i,j)
    if( i - 1 >= 0 and finalimg[i-1][j] == 1) :
        info[i][j][1] = right(i,j)
        info[i][j][3] = info[i-1][j][3] + 1
    else :
        info[i][j][1] = right(i,j)
    return 0
def BFS(x,y):
    # print(x," ",y)
    q = [[x,y]]
    visited[x][y] = 1
    centre = [x,y]
    minstd = 1000000
    while(len(q) != 0):
        i,j = q[0][0],q[0][1]
        q.pop(0)
        directions = [[i-1,j-1],[i,j-1],[i+1,j-1],[i-1,j],[i+1,j],[i-1,j+1],[i,j+1],[i+1,j+1]]
        for a in directions:
            if(a[0] < 0 or a[0] > len(finalimg) - 1  or a[1] < 0 or a[1] > len(finalimg[0]) - 1):
                continue
            else:
                if (finalimg[a[0]][a[1]] == 1 and visited[a[0]][a[1]] == 0):
                    q.append(a)
                    visited[a[0]][a[1]] = 1
                    tempstd = np.std(info[a[0]][a[1]])
                    if( tempstd < minstd):
                        minstd = tempstd 
                        centre[0] = i
                        centre[1] = j
    return [centre,max(info[centre[0]][centre[1]]),np.mean(info[centre[0]][centre[1]]),min(info[centre[0]][centre[1]]),info[centre[0]][centre[1]]]

row  = len(finalimg)
column = len(finalimg[0])
for i in range(row) :
    for j in range(column) :
        if(finalimg[i][j] == 1) :
            fill(i,j)
data = []
for i in range(len(finalimg)) :
    for j in range(len(finalimg[0])) :
        if(finalimg[i][j] == 1 and visited[i][j] == 0):
            data.append(BFS(i,j))

data = np.array(data)
df = pd.DataFrame()

n = len(data)
rowindex = []
columnindex = []

for i in range(n) :
    rowindex.append(data[:,0][i][0])
    columnindex.append(data[:,0][i][1])
    
df["Row"] = rowindex
df["Column"] = columnindex
df["meanRadius"] = data[:,2]
df["directionRadius"] = data[:,4]
df["maxRadius"] = data[:,1]
df["minRadius"] = data[:,3]
# print(df)
df.to_csv('DATAcsv.csv')
import matplotlib.pyplot as plt
plt.hist(df["meanRadius"])
plt.show()
