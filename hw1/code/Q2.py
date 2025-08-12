from  Q3 import inverse_fft as q3_inv_fft

from scipy.special import jn, jn_zeros
import numpy as np
import matplotlib.pyplot as plt
import cv2


def generate_polar_image(n, m, size=256):
    r = np.linspace(0, 1, size)
    theta = np.linspace(0, 2 * np.pi, size)
    R, Theta = np.meshgrid(r, theta)

    # Bessel function part
    zero = jn_zeros(n, m)[-1]
    radial = jn(n, zero * R)

    # Angular part
    if n == 0:
        img = radial
    else:
        img = radial * np.cos(n * Theta)  # Or np.sin(n*Theta)

    # Convert from polar to cartesian for display
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Interpolate into Cartesian grid for display
    from scipy.interpolate import griddata
    x_lin = np.linspace(-1, 1, size)
    y_lin = np.linspace(-1, 1, size)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    points = np.vstack((X.flatten(), Y.flatten())).T
    values = img.flatten()
    img_cart = griddata(points, values, (X_grid, Y_grid), method='cubic', fill_value=0)

    return img_cart


def show_and_save_polar_image(n, m, size=256,save=0,filename='polar_image.png',title=0):
    # Generate the polar image using the given parameters
    img = generate_polar_image(n, m, size)

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', extent=[-1, 1, -1, 1])
    if title: plt.title(f"Polar Image: n={n}, m={m}")
    plt.axis('off')  # Hide the axis for better visualization

    # Save the image
    if save: plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # Show the image
    plt.show()


def load_image(file_path):
    # Load the polar image
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


def show_image(img):
    plt.imshow(img, cmap='gray', extent=[-1, 1, -1, 1])


def compute_fft(image):
    # Compute 2D FFT
    fft_image = np.fft.fft2(image)
    # Shift zero-frequency component to the center
    fft_image_shifted = np.fft.fftshift(fft_image)

    return fft_image_shifted


def visualize_fft(fft_img):
    # Visualize the magnitude spectrum
    plt.imshow(np.log(1 + np.abs(fft_img)), cmap='gray')
    plt.title("Frequency Magnitude Spectrum")
    plt.show()


def build_lpf(img, alpha=0.05, filter_type='circular'):
    """
    Build a Low-Pass Filter (LPF) based on the specified type and alpha value.
    :param img: 2D image
    :param alpha: Fraction of the frequencies to keep (0 < alpha < 1)
    :param filter_type: 'x', 'y', or 'circular'
    :return: 2D binary mask
    """
    H, W = img.shape
    cx, cy = W // 2, H // 2

    lpf = np.zeros((H, W), dtype=np.uint8)

    if filter_type == 'x':
        kx = int((alpha * H * W) / (2 * H))
        lpf[:, cx - kx:cx + kx + 1] = 1

    elif filter_type == 'y':
        ky = int((alpha * H * W) / (2 * W))
        lpf[cy - ky:cy + ky + 1, :] = 1

    elif filter_type == 'o':
        r_max = np.sqrt((alpha * H * W) / np.pi)
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        lpf[dist <= r_max] = 1

    else:
        raise ValueError("Invalid filter type. Choose 'x', 'y', or 'circular'.")

    return lpf


def apply_lpf_to_fft(fft_image, filter_mask):
    """
    Apply the Low-Pass Filter (LPF) mask to the shifted FFT image.
    :param fft_image: The 2D FFT of the image, shifted (after np.fft.fftshift)
    :param filter_mask: The low-pass filter mask (2D array) to apply to the FFT

    :return: The filtered FFT image
    """
    # Apply the filter by element-wise multiplication
    filtered_fft = fft_image * filter_mask

    return filtered_fft


def inverse_fft(fft_spectrum):
    """
    Apply inverse FFT to the FFT spectrum to reconstruct an image.
    Generally used after filtering the spectrum.
    :param fft_spectrum: An FFT spectrum

    :return: The inverse FFT result, which is the an image
    """
    # Shift back the zero-frequency component to the original position
    fft_shifted_back = np.fft.ifftshift(fft_spectrum)

    # Apply inverse 2D FFT to get the filtered image
    filtered_image = np.fft.ifft2(fft_shifted_back)

    # Take the real part (since FFT result can be complex)
    filtered_image = np.abs(filtered_image)

    return filtered_image


def show_image_and_spectrum(image):
    """
    receives an image and shows it and its FFT spectrum
    :param image:
    """
    fft_shifted = compute_fft(image)

    # designed to show polar filter image and its Spectrum
    plt.figure(figsize=(15, 7))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.axis('off')

    # Magnitude spectrum
    plt.subplot(1, 2, 2)
    magnitude = np.log(1 + np.abs(fft_shifted))
    plt.imshow(magnitude, cmap='gray')
    plt.title("FFT Magnitude Spectrum")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def process_low_pass_filters(image, alpha,display=0):
    fft_image = compute_fft(image)

    x_result = apply_lpf_to_fft(fft_image, build_lpf(image, alpha,'x'))
    y_result = apply_lpf_to_fft(fft_image, build_lpf(image, alpha,'y'))
    circ_result = apply_lpf_to_fft(fft_image, build_lpf(image, alpha,'o'))

    # Show results
    titles = ["X-axis LPF", "Y-axis LPF", "Circular LPF"]
    images = [x_result, y_result, circ_result]

    if display:
        plt.figure(figsize=(15, 10))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(inverse_fft(images[i]), cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return q3_inv_fft(images)


def show_lpf_filters(img, alpha):
    # Show results
    plt.figure(figsize=(15, 4))
    titles = ["X-axis LPF", "Y-axis LPF", "Circular LPF"]
    images = [build_lpf(img,alpha,'x'),
              build_lpf(img,alpha,'y'),
              build_lpf(img,alpha,'o')]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_applied_lpf_fft(fft_img, alpha):
    # Show results
    plt.figure(figsize=(15, 4))
    titles = ["X-LPF spectrum", "Y-LPF spectrum", "O-LPF spectrum"]
    images = [apply_lpf_to_fft(fft_img, build_lpf(fft_img, alpha, 'x')),
              apply_lpf_to_fft(fft_img, build_lpf(fft_img, alpha, 'y')),
              apply_lpf_to_fft(fft_img, build_lpf(fft_img, alpha, 'o'))]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.log(1 + np.abs(images[i])), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def compute_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)