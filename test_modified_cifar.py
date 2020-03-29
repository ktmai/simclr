import pytest


@pytest.fixture
def tanda_data_loader(
    tanda_data_path="/home/amavorpa/learning_to_self_supervise/eg_batches/",
):
    from modified_cifar import CIFAR10_TANDA

    dataloader = CIFAR10_TANDA(root=tanda_data_path)
    return dataloader


def test_image_appearance(tanda_data_loader):
    import matplotlib.pyplot as plt

    item1, item2 = tanda_data_loader.__getitem__(index=0)
    plt.imshow(item1)
    plt.title("Not Augmented")
    plt.show()
    plt.close()

    plt.imshow(item2)
    plt.title("Augmented")
    plt.show()
    plt.close()
