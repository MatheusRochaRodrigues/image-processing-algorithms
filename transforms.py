import os

import numpy as np
from hsi_rgb_conv import hsi2rgb, rgb2hsi


import matplotlib.pyplot as plt

from common import read_img, save_img
import cv2

# Negativo
def negative(image):
    # TODO: Retorna o negativo fotografico de image
    
    if len(image.shape) == 2:  # Imagem em escala de cinza
        output = 255 - image

    if(len(image.shape) == 3):
        channelB, channelG, channelR = cv2.split(image)

        channelB = 255 - channelB #B
        channelG = 255 - channelG #G
        channelR = 255 - channelR #R

        output = np.concatenate((channelB[:, :, np.newaxis], 
                                 channelG[:, :, np.newaxis], 
                                 channelR[:, :, np.newaxis]), axis=2)
                                 

    return output



# Equalizacao de histogramas
def histeq(img):
    # Ajusta o brilho de uma imagem monocromatica
    hsiImgOriginal = rgb2hsi(img).astype('uint8')#.astype('float')
    hsiImg = hsiImgOriginal[:,:,2]
    
    histograma = np.zeros(256, dtype=np.float32)
    h, w = hsiImg.shape
    for i in range(h):
        for j in range(w):
            histograma[hsiImg[i, j]] += 1

    p = h * w
    histograma /= p
    
    histograma = cumsum(histograma)

    histograma *= 255
    histograma = np.round(histograma)

    histograma = histograma.astype(np.uint8)

    imagemTratada = np.zeros_like(hsiImg)
    for i in range(h):
        for j in range(w):
            imagemTratada[i, j] = histograma[hsiImg[i, j]]  #cuidado


    imagemTratada = np.clip(imagemTratada, 0, 255)
    hsiImgFinal = imagemTratada.astype('uint8')


    hsiImgOriginal[:,:,2] = hsiImgFinal
    hsiImgFinal = hsi2rgb(hsiImgOriginal)

    output = hsiImgFinal
    return output



def cumsum(vetor):
    cumsum = []
    acumulador = 0
    for elemento in vetor:
        acumulador += elemento
        cumsum.append(acumulador)
    return np.array(cumsum) 


#main

#3.2
img = read_img("lenacolor.jpg", False)
save_img(negative(img), "resultados/3.2_Ativ.jpg")

#3.6
img1 = read_img("Img1.png", False)
img2 = read_img("cat_puppy.jpg", False)
img1 = histeq(img1)
img2 = histeq(img2)

# save_img(img2, "resultados/Histo1.jpg")


plt.figure(figsize=(10, 5))  

plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, 1ª imagem
plt.imshow(img1, cmap='gray')
plt.title('Imagem 1')
plt.axis('off')  # Remove os eixos

# Exibir imagem equalizada
plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, 2ª imagem
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Imagem 2')
plt.axis('off')  # Remove os eixos

# Mostrar as imagens
plt.tight_layout()
plt.show()