import heapq
import pandas as pd

class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def compute_frequencies(pmf_dict, scale=1000000):
    """
    Converts a probability mass function (PMF) dictionary to frequency counts suitable for building a Huffman tree.

    Args:
        pmf_dict (dict): A dictionary mapping symbols (e.g., pixel values 0-255) to probabilities.

    Returns:
        list of tuples: Each tuple is (symbol, frequency), where frequency is an integer proportional to its probability.
    """
    return [(symbol, int(prob * scale)) for symbol, prob in pmf_dict.items() if prob > 0]


def build_huffman_tree(frequencies):
    """
       Builds a Huffman tree from the given symbol frequencies using a priority queue.

       Args:
           frequencies (list of tuples): Each tuple is (symbol, frequency).

       Returns:
           Node: The root of the built Huffman tree.
    """
    heap = [Node(symbol=sym, freq=freq) for sym, freq in frequencies]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]  # Root node


def generate_codebook(huffman_root):
    """
    Traverses the Huffman tree and assigns binary codes to each symbol.

    Args:
        huffman_root (HuffmanNode): The root of the Huffman tree.

    Returns:
        dict: A dictionary mapping symbols to binary string codes.
    """
    codebook = {}

    def traverse(node, path=""):
        if node.symbol is not None:
            codebook[node.symbol] = path
            return
        traverse(node.left, path + "0")
        traverse(node.right, path + "1")

    traverse(huffman_root)
    return codebook


def build_huffman(pmf_dict):
    """
        High-level function that takes a PMF and returns a Huffman codebook.

        Args:
            pmf_dict (dict): A dictionary mapping symbols to probabilities.

        Returns:
            dict: A dictionary mapping each symbol to its Huffman code (binary string).
    """
    freqs = compute_frequencies(pmf_dict)
    root = build_huffman_tree(freqs)
    return generate_codebook(root)


def build_codebooks(imgs_pmf_df):
    """
    Each entry in codebooks will map image index → Huffman codebook.
    """
    codebooks = {}
    for idx, row in imgs_pmf_df.iterrows():
        pmf_dict = row.to_dict()
        codebooks[idx] = build_huffman(pmf_dict)
    return codebooks
