import numpy as np
from matplotlib import pyplot as plt
import cv2


# ----- ATENÇAO - o senhor nao pediu para normalizar os valores mas como na questao da funcao save_img o senhor pede pra reescalar
# no intervalo de 0-255 acreditei que o senhor esqueceu de pedir pra normalizar essa imagem na função ent fiz uma booleana
# que normaliza se ela for colocada como VERDADEIRA

def read_img(path, grayscale=True, normalize = False):
    # Lê a imagem do arquivo especificado por path.
    # O argumento grayscale especifica se a imagem deve ser retornada
    # em tons de cinza ou colorida (BGR).
    
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(path)

    if normalize:
        img = img.astype(np.float32)
        img /= 255.0
    
    return img.astype(np.float32)



def save_img(img, path):
    # reescala o conteúdo de img para valores entre 0 e 255
    # e salva no caminho (path) especificado.
    img_rescaled = img
    if img.max() <= 1.0:
        img_rescaled = (img * 255.0).astype(np.uint8)

    img_rescaled.astype(np.uint8)
    cv2.imwrite(path, img_rescaled)


    print(path, "está salva!")
    


save_img(read_img("lenacolor.jpg", True, True), "resultados/2.2_Teste.jpg")

#2.3
img = read_img("lenagray.jpg")

h,w = img.shape
J = np.zeros((256, 256), dtype=np.float32)

J[:, :128] = img[:, 128:]
J[:, 128:] = img[:, :128]

save_img(J, "resultados/2.3_Ativ.jpg")

img = read_img("lenacolor.jpg", False)

#2.4
h, w, c = img.shape

J = np.zeros((h,w,c), dtype=np.uint8)

#0 -B  
#1 -G
#2 -R

J[:, :, 0] = img[:, :, 1]
J[:, :, 1] = img[:, :, 2] 
J[:, :, 2] = img[:, :, 0]

save_img(J, "resultados/2.4_Ativ.jpg")

print("sucess")