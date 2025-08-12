import Q2
from scipy.special import jn, jn_zeros
import numpy as np
import matplotlib.pyplot as plt
import cv2


def resize_image(img_from,img_to):
    return cv2.resize(img_from, (img_to.shape[1], img_to.shape[0]))
#
def cast_uint8(img):
    # Ensure image is in uint8 format
    return img.astype(np.uint8)

def display_images(img_arr,titles=None):
    n = len(img_arr)
    plt.figure(figsize=(15, 7))

    for i in range(n):
        # Display the images
        plt.subplot(1, n, 1+i)
        plt.imshow(img_arr[i], cmap='gray')
        if titles: plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def display_ffts(fft_arr, titles=None):
    n = len(fft_arr)
    plt.figure(figsize=(15, 7))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(np.log(1 + np.abs(fft_arr[i])), cmap='gray')
        if titles: plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()



def get_phase_fft(img):
    return np.angle(Q2.compute_fft(img))

def get_amplitude_fft(img):
    return np.abs(Q2.compute_fft(img))

def combine_amp_phase(amp,phase):
    return amp * np.exp(1j * phase)

def get_rand_amp(amp):
    return np.random.uniform(0, np.max(amp), amp.shape)

def get_rand_phase(phase):
    return np.random.uniform(-np.pi, np.pi, phase.shape)

def inverse_fft(fft_arr):
    """
    Apply inverse FFT to the FFT spectrum to reconstruct an image.
    :param fft_arr: An FFT spectrum

    :return: The inverse FFT result, which is an image
    """
    out_arr = []

    for fft in fft_arr:
        # Shift back the zero-frequency component to the original position
        fft_shifted_back = np.fft.ifftshift(fft)

        # Apply inverse 2D FFT to get the filtered image
        image = np.fft.ifft2(fft_shifted_back)

        # Take the real part (since FFT result can be complex)
        image = np.abs(image)
        out_arr.append(image)

    return out_arr




