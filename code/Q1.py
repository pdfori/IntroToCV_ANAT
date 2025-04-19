import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image in grayscale
def load_image_grey(image_path = '../given_data/boat.jpg' ):
    return  cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create figure with subplots
def plot_image_and_hist(img):
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Grayscale Image')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
    ax2.set_title('Histogram')
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def negative_img(img):
    """
    Perform negative on the given img.
    :param img: An input grayscale image - ndarray of uint8 type.
    :return:
        neg_img: An output grayscale image after applying negative -
                      uint8 ndarray of size [H x W x 1]
    """
    # ====== YOUR CODE: ======
    neg_img = 255 - img
    # ========================
    return neg_img

def contrast_enhancement(img: np.ndarray):
    """
    Perform contrast enhancment on the given img.
    :param img: An input grayscale image - ndarray of uint8 type.
    :return:
        contrast_enhanced_img: An output grayscale image after applying contrast enhancment -
                      uint8 ndarray of size [H x W x 1]
    """
    # ====== YOUR CODE: ======
    a = np.min(img)
    b = np.max(img)

    # Avoid division by zero in case all pixels have the same value
    if a == b:
        contrast_enhanced_img = np.zeros_like(img, dtype=np.uint8)
    else: # Apply the contrast enhancement formula
        contrast_enhanced_img = np.uint8((img.astype(np.float32) - a) / (b - a) * 255)
    # ========================
    return contrast_enhanced_img

def gamma_correction(img, gamma):
    """
    Perform gamma correction on a grayscale image.
    :param img: An input grayscale image - ndarray of uint8 type.
    :param gamma: the gamma parameter for the correction.
    :return:
        gamma_img: An output grayscale image after gamma correction -
                   uint8 ndarray of size [H x W x 1].
    """
    # ====== YOUR CODE: ======
    # Normalize image to range [0,1]
    img_norm = img / 255.0
    # Apply gamma correction
    corrected = np.power(img_norm, gamma)
    # Rescale back to [0,255]
    gamma_img = np.uint8(corrected * 255)
    # ========================

    return gamma_img

def apply_average_filter(img: np.ndarray, dim: int) -> np.ndarray:
    """
    Apply an average filter to a grayscale image using an NxN kernel.

    :param img: Input grayscale image (uint8 ndarray).
    :param dim: Size of the filter kernel (must be an odd integer).
    :return: Filtered image (uint8 ndarray).
    """
    kernel = np.ones((dim, dim), dtype=np.float32) / (dim * dim)
    return cv2.filter2D(img, -1, kernel)

def apply_median_filter(img: np.ndarray, ksize: int = 9):
    """
    Apply median filter to a grayscale image and plot the result and its histogram.

    :param img: Input grayscale image (uint8 ndarray).
    :param ksize: Size of the median filter kernel (must be an odd integer).
    :return: Filtered image (uint8 ndarray).
    """
    return cv2.medianBlur(img, ksize)


if __name__=="__main__":
    image_grey = load_image_grey('given_data/boat_brightened.jpg')
    enhanced  = contrast_enhancement(image_grey)
    plot_image_and_hist(enhanced)


