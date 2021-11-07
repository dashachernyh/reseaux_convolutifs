import cv2
import numpy as np
import random

def MaxPool(img):
    w, h = img.shape
    img_maxp = np.zeros([w - 1, h - 1], dtype='float')
    for i in range(h - 1):
        for j in range(w - 1):
            window = [img[i][j], img[i][j + 1], img[i + 1][j], img[i+1][j+1]]
            max_p = window[0]
            for k in range(1, 4):
                if max_p < window[k]:
                    max_p = window[k]
            img_maxp[i][j] = max_p
    return img_maxp


def Softmax(pix):
    e_x = np.exp(pix - np.max(pix))
    return e_x/e_x.sum()


# Лабораторная работа № 2

img = cv2.imread('cheval.jpg')
w, h, _ = img.shape
cv2.imshow('input', img)
# разбиваем на три канала
imgB, imgG, imgR = cv2.split(img)  # cv2.merge([imgB, imgG, imgR])

# слой 1
n, m = 3, 3
kernelB = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelG = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelR = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
imgB = cv2.filter2D(imgB, -1, kernelB)
imgG = cv2.filter2D(imgG, -1, kernelG)
imgR = cv2.filter2D(imgR, -1, kernelR)
img_1 = cv2.merge([imgB, imgG, imgR])
cv2.imshow('layer1', img_1)

# слой 2
kernelB = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelG = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelR = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])

imgB = cv2.filter2D(imgB, -1, kernelB)
imgG = cv2.filter2D(imgG, -1, kernelG)
imgR = cv2.filter2D(imgR, -1, kernelR)
img_2 = cv2.merge([imgB, imgG, imgR])
cv2.imshow('layer2', img_2)

# слой 3
kernelB = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelG = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelR = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
imgB = cv2.filter2D(imgB, -1, kernelB)
imgG = cv2.filter2D(imgG, -1, kernelG)
imgR = cv2.filter2D(imgR, -1, kernelR)
img_3 = cv2.merge([imgB, imgG, imgR])
cv2.imshow('layer3', img_3)

# слой 4
kernelB = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelG = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelR = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
imgB = cv2.filter2D(imgB, -1, kernelB)
imgG = cv2.filter2D(imgG, -1, kernelG)
imgR = cv2.filter2D(imgR, -1, kernelR)
img_4 = cv2.merge([imgB, imgG, imgR])
cv2.imshow('layer4', img_4)

# слой 5
kernelB = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelG = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
kernelR = np.array([[random.randint(-9, 9) for j in range(m)] for i in range(n)])
imgB = cv2.filter2D(imgB, -1, kernelB)
imgG = cv2.filter2D(imgG, -1, kernelG)
imgR = cv2.filter2D(imgR, -1, kernelR)
img_5 = cv2.merge([imgB, imgG, imgR])
cv2.imshow('layer5', img_5)

# нормализация

imgB = cv2.normalize(imgB, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
imgG = cv2.normalize(imgG, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
imgR = cv2.normalize(imgR, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# reLU
for i in range(h):
    for j in range(w):
       imgB[i][j] = max(0, imgB[i][j])
       imgG[i][j] = max(0, imgG[i][j])
       imgR[i][j] = max(0, imgR[i][j])

img_6 = cv2.merge([imgB, imgG, imgR])
cv2.imshow('relu', img_6)


# maxpool

imgB = MaxPool(imgB)
imgG = MaxPool(imgG)
imgR = MaxPool(imgR)

print(imgB.shape)
print(imgG.shape)
print(imgR.shape)

img_7 = cv2.merge([imgB, imgG, imgR])
cv2.imshow('max_pool', img_7)

# softmax
for i in range(h-1):
    for j in range(w-1):
       img_7[i][j] = Softmax(img_7[i][j])

cv2.imshow('softmax', img_7)

cv2.waitKey(0)
