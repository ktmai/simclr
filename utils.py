import matplotlib.pyplot as plt 
import numpy as np
import torchvision
import torch 

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(train_loader, temp_model_path):
    from transformer import Transformer

    with torch.no_grad():
        # Get a batch of training data
        cpu_model = Transformer()
        cpu_model.load_state_dict(torch.load(temp_model_path))

        data = next(iter(train_loader))[0]
        input_tensor = data.cpu()
        transformed_input_tensor = cpu_model.transform(data).cpu()

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
