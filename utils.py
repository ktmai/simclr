import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from RotationTransformer import RotationTransformer

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(train_loader, temp_model_path):
    with torch.no_grad():
        # Get a batch of training data
        cpu_model = RotationTransformer()
        cpu_model.load_state_dict(torch.load(temp_model_path))

        data = next(iter(train_loader))[0]
        input_tensor = data.cpu()
        augmented, _, not_augmented, _ = cpu_model(data)
        transformed_input = torch.cat([augmented, not_augmented], dim=0)
        transformed_input_tensor = transformed_input.cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor)
        )

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        axarr[1].imshow(out_grid)
        axarr[1].set_title("Transformed Images")

if __name__ == '__main__':
    main()
