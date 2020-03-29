import pytest


@pytest.fixture
def tanda_data_loader(
    tanda_data_path="/home/amavorpa/learning_to_self_supervise/eg_batches/",
):
    from modified_cifar import CIFAR10_TANDA

    dataloader = CIFAR10_TANDA(root=tanda_data_path)
    return dataloader



def test_sort_strings_based_on_digits(tanda_data_loader):
    string_list = ["a_sting_1113394-.npy", "1111_weird_2.npy", "string_23333333", " a a a +st 1, 1111+_2"]
    sorted_strings = tanda_data_loader.sort_strings_based_on_digits(string_list)
    true_sorted = [string_list[3], string_list[1], string_list[0], string_list[2]]
    assert true_sorted == sorted_strings



def test_image_appearance(tanda_data_loader):
    import matplotlib.pyplot as plt
    import numpy as np

    f, axarr = plt.subplots(2, 2)
    item1, item2, dummy_target = tanda_data_loader.__getitem__(index=0)
    assert np.array_equal(item1, item2) == False

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(item1.reshape(32, 32, 3))
    axarr[1].imshow(item2.reshape(32, 32, 3))

    plt.show()
    plt.close()
