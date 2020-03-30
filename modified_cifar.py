"""Modified version of CIFAR-10 used for Siamese network
"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader


class CIFAR10_TANDA(Dataset):
    def __init__(self, root):
        super(CIFAR10_TANDA, self).__init__()

        (
            transformed_batches,
            not_transformed_batches,
        ) = self.get_transformed_and_not_batches(root)
        self.not_transformed_data = self.load_data(transformed_batches)
        self.transformed_data = self.load_data(not_transformed_batches)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img1, img2, target) where img1 and img2 are identical
            samples with random transformations target is index of the target
            class.
        """
        self.shuffle_data()
        img1, img2 = self.not_transformed_data[index], self.transformed_data[index]
        target = 1  ## dummy target not actually used

        return img1, img2, target

    def __len__(self):
        return len(self.transformed_data)

    def shuffle_data(self):
        # shuffle hack TODO fix this
        import random

        zipped = list(zip(self.not_transformed_data, self.transformed_data))
        random.shuffle(zipped)
        self.not_transformed_data, self.transformed_data = zip(*zipped)

    def get_and_separate_paths(self, directory):
        import glob

        all_paths = glob.glob(directory + "*npy")
        not_transformed_paths = [x for x in all_paths if "not" in x]
        transformed_paths = [x for x in all_paths if "not" not in x]
        assert self.common_member(transformed_paths, not_transformed_paths) == False
        return transformed_paths, not_transformed_paths

    def get_transformed_and_not_batches(self, root):
        transformed_paths, not_transformed_paths = self.get_and_separate_paths(
            directory=root
        )
        transformed_paths = self.sort_strings_based_on_digits(transformed_paths)
        not_transformed_paths = self.sort_strings_based_on_digits(not_transformed_paths)
        return transformed_paths, not_transformed_paths

    def load_data(self, batch_paths):
        import numpy as np

        data_arr = []
        for a_path in batch_paths:
            batch = np.load(a_path)
            for img in batch:
                data_arr.append(img.reshape(3, 32, 32))

        data_arr = np.array(data_arr)

        return data_arr

    def sort_strings_based_on_digits(self, string_list):
        import re

        string_list = sorted(string_list, key=lambda x: re.findall(r"\d+", x)[0])
        return string_list

    def common_member(self, a, b):
        a_set = set(a)
        b_set = set(b)
        if a_set & b_set:
            return True
        else:
            return False
