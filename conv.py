import cv2
import numpy as np
import random


def conv(image, kernel):
     imgB, imgG, imgR = cv2.split(image)
     kerB, kerG, kerR = cv2.split(kernel)
# прменили свертку послойно и просуммировали поканально = скаляроное прозведение
     resB = cv2.filter2D(imgB, -1, kerB)
     resG = cv2.filter2D(imgG, -1, kerG)
     resR = cv2.filter2D(imgR, -1, kerR)
     res = resB + resG + resR
     return res


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
w, h = img.shape[:2]

# создаем фильтр 3х3х3
n, m, k = 3, 3, 3
kernel_1 = np.array([[[random.randint(-9, 9)for p in range(k)]for j in range(m)]for i in range(n)])

kernel_2 = np.array([[[random.randint(-9, 9)for p in range(k)]for j in range(m)]for i in range(n)])

kernel_3 = np.array([[[random.randint(-9, 9)for p in range(k)]for j in range(m)]for i in range(n)])

kernel_4 = np.array([[[random.randint(-9, 9)for p in range(k)]for j in range(m)]for i in range(n)])

kernel_5 = np.array([[[random.randint(-9, 9)for p in range(k)]for j in range(m)]for i in range(n)])

# сверточный слой
convolution = [kernel_1, kernel_2, kernel_3, kernel_4, kernel_5]

# выход 5-ти канальный
zero_img = np.zeros(shape=(w, h))

out = np.array([zero_img for i in range(5)])
for i in range(5):
    out[i] = conv(img, convolution[i])

# нормализация
for i in range(5):
    out[i] = cv2.normalize(out[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

for i in range(5):
    cv2.imshow(str(i), out[i])


# reLU
for m in range(5):
    for i in range(h):
        for j in range(w):
           out[m][i][j] = max(0, out[m][i][j])

for i in range(5):
    cv2.imshow(str(i)+"r", out[i])


# max-pooling c шагом 1
zero_img = np.zeros(shape=(w-1, h-1))
out_res = np.array([zero_img for i in range(5)])
for m in range(5):
    out_res[m] = MaxPool(out[m])

for i in range(5):
    cv2.imshow(str(i)+"m", out_res[i])

img_5 = cv2.merge([out_res[0], out_res[1], out_res[2], out_res[3], out_res[4]])

# softmax

for i in range(h-1):
    for j in range(w-1):
        img_5[i][j] = Softmax(img_5[i][j])

out_res[0], out_res[1], out_res[2], out_res[3], out_res[4] = cv2.split(img_5)

for i in range(5):
    cv2.imshow(str(i)+"s", out_res[i])

print(out_res.shape)
cv2.waitKey(0)