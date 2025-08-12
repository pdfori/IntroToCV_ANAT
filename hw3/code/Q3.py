import cv2
from matplotlib import pyplot as plt


def poisson_noisy_image(X, a):
    """
    Creates a Poisson noisy image.
    :param X: The Original image. np array of size [H x W] and of type uint8.
    :param a: number of photons scalar factor
    :return:
        Y: The noisy image. np array of size [H x W] and of type uint8.
    """
    X = X.astype(float)                  # Convert to float
    photons = X * a                      # Convert gray levels to photon counts
    noisy_photons = np.random.poisson(photons)  # Apply Poisson noise
    Y = noisy_photons / a                # Convert back to gray levels
    Y = np.clip(Y, 0, 255)               # Clip to valid range
    Y = Y.astype(np.uint8)               # Convert back to uint8

    return Y


import numpy as np
from scipy.signal import convolve2d


def denoise_by_l2(Y, X, num_iter, lambda_reg):
    """
    L2 image denoising using steepest descent.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """

    # ====== YOUR CODE: ======
    h, w = Y.shape
    y_vec = Y.flatten(order='F')  # column vector
    xk = y_vec.copy()

    Err1 = np.zeros(num_iter)
    Err2 = np.zeros(num_iter)

    # Laplacian kernel
    D_kernel = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])

    for k in range(num_iter):
        # Compute gradient Gk
        xk_mat = np.reshape(xk, (h, w), order='F')
        Dx = cv2.filter2D(xk_mat, -1, D_kernel, borderType=cv2.BORDER_REFLECT)
        DTDx = cv2.filter2D(Dx, -1, D_kernel, borderType=cv2.BORDER_REFLECT)
        grad = (xk - y_vec) + lambda_reg * DTDx.flatten(order='F')

        # Step size mu_k
        grad_mat = np.reshape(grad, (h, w), order='F')
        Dg = cv2.filter2D(grad_mat, -1, D_kernel, borderType=cv2.BORDER_REFLECT)
        DTDg = cv2.filter2D(Dg, -1, D_kernel, borderType=cv2.BORDER_REFLECT)
        DTDg_vec = DTDg.flatten(order='F')

        numerator = np.sum(grad ** 2)
        denominator = np.sum(grad * (grad + lambda_reg * DTDg_vec)) + 1e-8
        mu_k = numerator / denominator

        # Update
        xk = xk - mu_k * grad

        # Errors
        xk_mat = np.reshape(xk, (h, w), order='F')
        Dxk = cv2.filter2D(xk_mat, -1, D_kernel, borderType=cv2.BORDER_REFLECT)
        Err1[k] = np.sum((xk - y_vec) ** 2) + lambda_reg * np.sum(Dxk ** 2)
        Err2[k] = np.sum((xk - X.flatten(order='F')) ** 2)

    Xout = np.reshape(xk, (h, w), order='F')
    Xout = np.clip(Xout, 0, 255).astype(np.uint8)

    return Xout, Err1, Err2
    # ========================


def display_denoising(Xrestored, Err1, Err2, title="", label1='Err1 (vs noisey)', label2='Err2 (vs Orig)'):
    # Display
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.imshow(Xrestored, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Log Error vs Iteration")
    plt.plot(np.log10(Err1), label=label1)
    plt.plot(np.log10(Err2), label=label2)
    plt.xlabel('Iteration')
    plt.ylabel('log10(Error)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



def denoise_by_TV(Y, X, num_iter, lambda_reg, epsilon0):
    """
    TV image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :param: epsilon0: small scalar for numerical stability
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    # ====== YOUR CODE: ======
    Xk = Y.copy()
    mu = 150 * epsilon0

    Err1 = np.zeros(num_iter)
    Err2 = np.zeros(num_iter)

    for k in range(num_iter):
        # Compute gradients
        grad_y, grad_x = np.gradient(Xk)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + epsilon0**2)
        gx = grad_x / grad_magnitude
        gy = grad_y / grad_magnitude

        # Compute divergence of (gx, gy)
        div_gx = np.gradient(gx, axis=1)
        div_gy = np.gradient(gy, axis=0)
        div = div_gx + div_gy

        # Compute update term Uk
        Uk = 2 * (Y - Xk) + lambda_reg * div

        # Gradient descent step
        Xk = Xk + (mu / 2) * Uk

        # Compute errors
        Err1[k] = np.sum((Xk - Y)**2) + lambda_reg * np.sum(np.sqrt(grad_x**2 + grad_y**2 + epsilon0**2))
        Err2[k] = np.sum((Xk - X)**2)

    Xout = Xk
    # ========================
    return Xout, Err1, Err2