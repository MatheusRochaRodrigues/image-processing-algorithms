import os

import numpy as np

from common import read_img, save_img

import cv2


def gaussian_high_pass(img, D0):
    # Entrada: imagem e a frequencia de corte D0
    # Saida: imagem filtrada por um fitro gaussiano passa-altas no dominio da frequencia
    H, W = img.shape
    Hp = 2 * H
    Wp = 2 * W

    padded_img = cv2.copyMakeBorder(img, top=0, left=0, right=W, bottom=H, borderType=cv2.BORDER_CONSTANT, value=[0])

    #deslocando pro centro a imagem
    x = np.arange(Hp)
    y = np.arange(Wp)
    X, Y = np.meshgrid(y, x)
    desloca = (-1)**(X + Y)
    padded_shifted_img = padded_img * desloca

    Fernel = np.fft.fft2(padded_shifted_img)

    #criamos o filtro no dominio da frequecnai
    u = np.concatenate((np.arange(W, 0, -1), np.arange(W)))
    v = np.concatenate((np.arange(H, 0, -1), np.arange(H)))
    U, V = np.meshgrid(u, v) 
    D = np.sqrt(U**2 + V**2)

    Filtro = 1 - np.exp(-D**2/(2*D0**2))    

    #multiplicaçao ponto a ponto no dominio de furrier
    F_filtered = Filtro * Fernel

    #transformada inversa e descentraliza
    img_filtered = np.real(np.fft.ifft2(F_filtered))
    img_filtered = img_filtered * desloca

    output = img_filtered[:H,:W]   #tiro as bordas
    output[output < 0] = 0
    output[output > 255] = 255

    return output

    

def steerable_filter(image, angles=(np.pi * np.arange(6, dtype=float) / 6)):
    # Dado uma lista de angulos usados como alpha, retorna
    # uma lista contendo as imagens baseada na formula do pdf.
    # Entrada - imagem: H x W
    # Ângulos: uma lista de escalares
    # Saída: uma lista de imagens H x W

    # TODO: Use convolve() para completar a funcao
    Mag, Gx, Gy = sobel(image)
    
    ImagesSteerable = []
    for angle in angles:
        ImagesSteerable.append( (np.cos(angle) * Gx) + (np.sin(angle) * Gy) )

    output = ImagesSteerable

    return output


def sobel(image):
    # Retorna a magnitude do gradiente da imagem de entrada
    # Entrada- imagem: H x W
    # Saída - grad_magnitude: H x W

    # TODO: defina os kernels de sobel sx e sy
    sx = np.array([ [-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])  # 1 x 3
    
    sy = np.array([ [-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]])  # 3 x 1

    Ix = convolve(image, sx)
    Iy = convolve(image, sy)

    # TODO: Use Ix, Iy para calcular a magnitude do gradiente
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    grad_magnitude = np.uint8(grad_magnitude)

    return grad_magnitude, Ix, Iy






def image_patches(image, patch_size=(16, 16)):
    # Divide a imagem de entrada em partes de tamanho
    # patch_size e as retorna normalizadas em uma lista.
    # image: H x W
    # patch_size: uma tupla de escalares (M, N)
    # retorno: uma lista de imagens de tamanho M x N
    # Use numpy slicing para completar a funcao.

    patchList = []
    h = image.shape[0]
    w = image.shape[1]
    
    if len(image.shape) == 3:
        for i in range(0 , h - patch_size[0], patch_size[0]) :
            for j in range(0 , w - patch_size[1], patch_size[1]) :
                patchList.append(image[i:i+patch_size[0], j:j+patch_size[1], :])
            
    if len(image.shape) == 2:
            for i in range(0 , h - patch_size[0], patch_size[0]) :
                for j in range(0 , w - patch_size[1], patch_size[1]) :
                    patchList.append(image[i:i+patch_size[0], j:j+patch_size[1]])
            

    output = patchList
    return output




def NormalizePatchs(patch):

    pt_normalize = []
    #----------Normalizando
    #calcula a media somando todos os elementos e dividindo pelo numero de elementos
    media = np.mean(patch)
    #calcula o desvio padrao que é o quanto o valor se desvia da media
    desvio = np.std(patch)

    for pt in patch:
        #Normalizando a partir dq começamos tornando o vlaor medio dos dados 0 subtraindo o patch pela media
        if(desvio == 0):
            #obter media zero pra centralizar
            pt_normalize.append(pt - media)


        #Disperção normalizada      dividindo o desvio a variancia se torna 1
        pt_normalize.append((pt - media)/desvio)

    return pt_normalize




def edge_detection(image):
    # Retorna a magnitude do gradiente da imagem de entrada
    # Entrada- imagem: H x W
    # Saída - grad_magnitude: H x W

    # TODO: defina os kernels kx e ky
    kx = np.array([     [0, -1, 0],
                        [0,  0, 0],
                        [0,  1, 0]])
    
    ky = np.array([     [0,  0, 0],
                        [-1, 0, 1],
                        [0,  0, 0]])
    
    # Ix = convolve(image, kx)
    # Iy = convolve(image, ky)
    
    # Kernel kx separável
    kxVertical = np.array   ([-1,   0,  1])
    kxHorizontal = np.array ( [0,  -1,  0])

    # Kernel ky separável
    kyVertical = np.array   ([0,   1,   0])
    kyHorizontal = np.array ([1,   0,  -1])

    Ix = convolveSep(image, kxHorizontal, kxVertical)
    Iy = convolveSep(image, kyHorizontal, kyVertical)


    kx = (1.0/2) * kx
    ky = (1.0/2) * ky


    # TODO: Use Ix, Iy para calcular a magnitude do gradiente
    grad_magnitude = np.sqrt((Ix**2) + (Iy**2))

    #normalizo pra faixa de 0 a 255
    grad_magnitude = 255 * (grad_magnitude - np.min(grad_magnitude)) / (np.max(grad_magnitude) - np.min(grad_magnitude))
    

    
    return grad_magnitude, Ix, Iy



def convolveSep(channel, kernel_horizontal, kernel_vertical):
    # np.rot90(kernel, 2)  #kernel é 90 graus rotacionado 2 vezes

    img_height, img_width = channel.shape
    
    # Aplica o filtro horizontal
    temp_image = np.zeros_like(channel)
    pad_width = kernel_horizontal.size // 2
    padded_channel = np.pad(channel, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    for i in range(img_height):
        for j in range(img_width):
            region = padded_channel[i, j:j + kernel_horizontal.size]
            temp_image[i, j] = np.sum(region * kernel_horizontal)
    
    # Aplica o filtro vertical
    output_image = np.zeros_like(channel)
    pad_height = kernel_vertical.size // 2
    padded_temp_image = np.pad(temp_image, ((pad_height, pad_height), (0, 0)), mode='constant', constant_values=0)
    
    for i in range(img_height):
        for j in range(img_width):
            region = padded_temp_image[i:i + kernel_vertical.size, j]
            output_image[i, j] = np.sum(region * kernel_vertical)
    
    return output_image


def convolve(channel, kernelM):
    # flipando no eixo horizontal e vertical oq resulta em algo equivalente a rotacionar em 180 graus o kernel para poder convolucionar com a imagem
    kernel = np.flipud(np.fliplr(kernelM))

    img_height, img_width = channel.shape       #image
    kernel_height, kernel_width = kernel.shape

    pad = kernel_height // 2

    # preencho as bordas
    padded_image  = np.pad(channel, pad, mode='constant', constant_values=0)
    output_image = np.zeros_like(channel)

    # para cada pixel
    for i in range(img_height):
        for j in range(img_width):

            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            
            output_image[i, j] = (np.sum(region * kernel))

    return output_image


#4.2
img = read_img("grace_hopper.png")

patch = image_patches(img, (16*1, 16*1))

pt = NormalizePatchs(patch)
#vamos colocar agora no intervalo de 0 a 255 ja q pode haver resultados negativos e tb estao normalizados para variancia igual a 1
ptNorm = ((pt - np.min(pt)) / (np.max(pt) - np.min(pt)) * 255).astype(np.uint8) # Converte para tipo uint8
    

save_img(ptNorm[0], f'resultados/patchs/patch1.png')
save_img(ptNorm[20], f'resultados/patchs/patch2.png')
save_img(ptNorm[81], f'resultados/patchs/patch3.png')


#4.6
#convolução
gaussianKernel= np.array([      [0.05854983, 0.09653235, 0.05854983],
                                [0.09653235, 0.15915494, 0.09653235],
                                [0.05854983, 0.09653235, 0.05854983]])

save_img(convolve(img,gaussianKernel), "resultados/4.6_Ativ.png")

#4.7
imgsobel = read_img("lenacolor.jpg")

MagK, kx, ky = edge_detection(imgsobel) 
save_img(kx, "resultados/4.7_Ativ_kx.png")
save_img(ky, "resultados/4.7_Ativ_ky.png")
save_img(MagK, "resultados/4.7_Ativ_Mag.png")


#4.8
# print("debug")
MagS, sX, sY = sobel(imgsobel)
# print(MagS)
save_img(sX, "resultados/4.8_Ativ_sobelHorizontal.png")
save_img(sY, "resultados/4.8_Ativ_sobelVertical.png")
save_img(MagS, "resultados/4.8_Ativ_sobel.png")


#1
steerable = steerable_filter(imgsobel)
save_img(steerable[0], "resultados/SterrableFilter/0.png")
save_img(steerable[1], "resultados/SterrableFilter/pi6.png")
save_img(steerable[2], "resultados/SterrableFilter/pi3.png")
save_img(steerable[3], "resultados/SterrableFilter/pi2.png")
save_img(steerable[4], "resultados/SterrableFilter/2pi3.png")
save_img(steerable[5], "resultados/SterrableFilter/5pi6.png")



#5.4
img = read_img("lenacolor.jpg")
img_gausFurrier = gaussian_high_pass(img, 20.0)
save_img(img_gausFurrier, "resultados/5.4_Ativ_FurrierGauss.png")
#5.5
img_gausFurrierAgucada = img_gausFurrier + img
img_gausFurrierAgucada[img_gausFurrierAgucada < 0] = 0
img_gausFurrierAgucada[img_gausFurrierAgucada > 255] = 255
save_img(img_gausFurrierAgucada, "resultados/5.5_Ativ_FurrierAgucadaGauss.png")

