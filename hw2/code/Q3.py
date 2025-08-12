import os
import pandas as pd
import numpy as np
from IPython.testing.plugin.pytest_ipdoctest import pytest_collect_file
from PIL import Image
from skimage import data
import matplotlib.pyplot as plt
import heapq
import huffman

def show_images_from_dir(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    n = len(files)

    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, fname in enumerate(sorted(files)):
        img = Image.open(os.path.join(directory, fname))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(fname)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def create_database_using_skimage(dir="../data"):
    # Create output directory
    os.makedirs(dir, exist_ok=True)

    # Target size image?
    target_size = (256, 256)

    # Load skimage images
    image_sources = {
        "camera": data.camera(),
        "coins": data.coins(),
        "moon": data.moon(),
        "page": data.page(),
        "text": data.text(),
        "clock": data.clock(),
        "chelsea_gray": data.chelsea()[..., 0],  # RGB -> use red channel
    }

    # Save all images as 256x256 grayscale PNGs
    for name, arr in image_sources.items():
        img = Image.fromarray(arr).convert("L")  # Ensure grayscale
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img.save(f"{dir}/{name}.png")

    print(f"Finished generating dataset. Saved all images to {dir}")


def compute_image_pmf(image):
    flat = image.flatten()
    counts = np.bincount(flat, minlength=256)
    pmf = counts / counts.sum()
    return pmf


def entropy(pmf):
    pmf = np.array(pmf)
    pmf = pmf[pmf > 0]  # avoid log(0)
    return -np.sum(pmf * np.log2(pmf))


def generate_img_pmf_dict(pmf_dict=None,wd="../data"):
    if pmf_dict is None:
        pmf_dict = {}
    else:
        pmf_dict = pmf_dict

    for filename in os.listdir(wd):
        if filename.endswith(".png"):
            img = np.array(Image.open(os.path.join(wd, filename)))
            pmf_dict[filename] = compute_image_pmf(img)

    pmf_df = pd.DataFrame.from_dict(pmf_dict, orient='index')
    pmf_df.columns = range(256)

    return pmf_df

def generate_img_entropy_df(entropy_dict=None, wd="../data"):
    if entropy_dict is None:
        entropy_dict = {}
    else:
        entropy_dict = entropy_dict

    for filename in os.listdir(wd):
        if filename.endswith(".png"):
            img = np.array(Image.open(os.path.join(wd, filename)))
            entropy_dict[filename] = entropy(compute_image_pmf(img))

    pmf_df = pd.DataFrame.from_dict(entropy_dict,columns=["entropy"], orient='index')

    return pmf_df

def entropy(pmf):
    pmf = np.array(pmf)
    pmf = pmf[pmf > 0]  # avoid log(0)
    return -np.sum(pmf * np.log2(pmf))


def compute_optimal_metrics_df(imgs_pmf_df, codebooks_df):
    """
    Computes expected code length and KL divergence for each image, based on the code of the separate images .

    Args:
        imgs_pmf_df (pd.DataFrame): Each row is the PMF of an image (shape: num_images x 256).
        avg_pmf (pd.Series): The average PMF across all images (length 256).
        codebooks_df (pd.DataFrame): Each column is the code length per symbol for one image (shape: 256 x num_images).

    Returns:
        pd.DataFrame: Contains expected code length and KL divergence for each image.
    """
    avg_pmf = imgs_pmf_df.mean(axis=0)

    codebooks_len_df = codebooks_df.applymap(lambda x: len(x) if isinstance(x, str) else np.nan)

    aligned_pmf = imgs_pmf_df.T.sort_index().T
    aligned_len = codebooks_len_df.fillna(0).sort_index().T

    expected_lengths = (aligned_pmf * aligned_len).sum(axis=1)


    kl_divergences = (
            imgs_pmf_df * np.log2(imgs_pmf_df / avg_pmf).replace([np.inf, -np.inf, np.nan], 0)
    ).sum(axis=1)

    return pd.DataFrame({
        "ExpectedLength": expected_lengths,
        "KL_Divergence": kl_divergences
    })

def compute_metrics_df(imgs_pmf_df):
    """
    Computes expected code length and KL divergence for each image, based on universal code.

    Args:
        imgs_pmf_df (pd.DataFrame): Each row is the PMF of an image (shape: num_images x 256).


    Returns:
        pd.DataFrame: Contains expected code length and KL divergence for each image.
    """
    avg_pmf = imgs_pmf_df.mean(axis=0)
    avg_pmf_df = pd.DataFrame(imgs_pmf_df.mean(axis=0)).rename(columns={0:"probability"})

    avg_codebook = pd.DataFrame.from_dict(huffman.build_huffman(avg_pmf), orient='index').rename(columns={0:"code"})


    # multiply histograms
    aligned_pmf = imgs_pmf_df.T.sort_index().T
    aligned_len = avg_codebook.applymap(lambda x: len(x) if isinstance(x, str) else 0).sort_index()


    expected_lengths = aligned_pmf.mul(aligned_len["code"],axis=1).sum(axis=1)


    kl_divergences = (
            imgs_pmf_df * np.log2(imgs_pmf_df / avg_pmf).replace([np.inf, -np.inf, np.nan], 0)
    ).sum(axis=1)

    return pd.DataFrame({
        "ExpectedLength": expected_lengths,
        "KL_Divergence": kl_divergences
    })


def expected_lengths_from_codebook_df(pmf_df, codebooks_df):
    """
    Computes the expected code length per image given a PMF DataFrame and a codebook DataFrame.

    Args:
        pmf_df (pd.DataFrame): Each row is the PMF of an image (shape: num_images x 256).
        codebooks_df (pd.DataFrame): Each column is a codebook (code string per grey level), shape: 256 x num_images.

    Returns:
        pd.Series: Expected code length per image (index matches pmf_df).
    """
    # Convert code strings to lengths
    code_len_df = codebooks_df.applymap(lambda x: len(x) if isinstance(x, str) else np.nan)

    # Ensure alignment and fill missing values
    aligned_pmf = pmf_df.T.sort_index().T
    aligned_len = code_len_df.fillna(0).sort_index().T

    # Compute expected lengths
    expected_lengths = (aligned_pmf * aligned_len).sum(axis=1)

    return expected_lengths


def plot_results(metrics_df,
                 x_axis="KL_Divergence",
                 y_axis="ExpectedLength",
                 xlabel="KL Divergence",
                 ylabel="ExpectedLength",
                 title="Compression vs Divergence"
                 ):
    """
    Plots KL divergence vs expected code length.

    Args:
        metrics_df (pd.DataFrame): Must contain 'KL_Divergence' and 'ExpectedLength' columns.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(metrics_df[x_axis], metrics_df[y_axis], color='blue')
    for idx, row in metrics_df.iterrows():
        plt.annotate(str(idx),
                     (row[x_axis], row[y_axis]),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='left', fontsize=8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

