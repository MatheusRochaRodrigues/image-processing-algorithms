import numpy as np
import cv2
def rgb2hsi(image: np.ndarray) -> np.ndarray:
    
    image = image.astype('float') / 255

    # Separate the individual channels
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Calculate Intensity
    intensity = (r + g + b) / 3

    # Calculate Saturation
    min_rgb = np.min(image, axis=2)
    saturation = 1 - (3 / (r + g + b + 1e-6)) * min_rgb

    # Calculate Hue
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    hue = np.arccos(num / (den + 1e-6))
    hue[b > g] = 2 * np.pi - hue[b > g]
    hue = hue / (2 * np.pi)  # Normalize hue to [0, 1]

    # Stack the H, S, and I channels
    hsi_image = np.stack((hue, saturation, intensity), axis=-1)

    return (hsi_image * 255).astype('uint8')




def hsi2rgb(image: np.ndarray) -> np.ndarray:
    # Normalize the H, S, and I values to the [0, 1] range
    h = image[:, :, 0] / 255.0 * 2 * np.pi  # Hue values in radians
    s = image[:, :, 1] / 255.0              # Saturation values
    i = image[:, :, 2] / 255.0              # Intensity values

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # H values in [0, 2*pi/3)
    mask = (h >= 0) & (h < 2 * np.pi / 3)
    b[mask] = i[mask] * (1 - s[mask])
    r[mask] = i[mask] * (1 + s[mask] * np.cos(h[mask]) / np.cos(np.pi / 3 - h[mask]))
    g[mask] = 3 * i[mask] - (r[mask] + b[mask])

    # H values in [2*pi/3, 4*pi/3)
    mask = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    h_adj = h[mask] - 2 * np.pi / 3
    r[mask] = i[mask] * (1 - s[mask])
    g[mask] = i[mask] * (1 + s[mask] * np.cos(h_adj) / np.cos(np.pi / 3 - h_adj))
    b[mask] = 3 * i[mask] - (r[mask] + g[mask])

    # H values in [4*pi/3, 2*pi)
    mask = (h >= 4 * np.pi / 3) & (h < 2 * np.pi)
    h_adj = h[mask] - 4 * np.pi / 3
    g[mask] = i[mask] * (1 - s[mask])
    b[mask] = i[mask] * (1 + s[mask] * np.cos(h_adj) / np.cos(np.pi / 3 - h_adj))
    r[mask] = 3 * i[mask] - (g[mask] + b[mask])

    # Combine the R, G, B channels and denormalize to [0, 255]
    rgb_image = np.stack((r, g, b), axis=-1) * 255.0
    rgb_image = np.clip(rgb_image, 0, 255).astype('uint8')

    return rgb_image
