import torch
import torch.nn as nn
from micromind import PhiNet
from micromind.networks.phinet import PhiNetConvBlock, SeparableConv2d
from torchsummary import summary

def remove_ModuleList(network, all_layers):
    """
    Recursively removes nn.ModuleList layers and adds all other layers to a list.

    :param network (nn.Module): The network or layer to process.
    :param all_layers (list): List to store all layers.
    """

    for layer in network.children():
        # If ModuleList layer, apply recursively to layers in ModuleList layer
        if isinstance(layer, nn.ModuleList):
            remove_ModuleList(layer, all_layers)
        else:  # if leaf node, add it to list
            all_layers.append(layer)

class PhiNet_v3(nn.Module):
    """PhiNet implementation. This model
    can be instantiated from a pretrained network."""

    def __init__(self, model, latent_layer_num = 0, device = "cpu",):
        """
        Initialize the model.

        :param pretrained: Path to the pretrained model.
        :param input_shape: Shape of the input data.
        :param alpha: Alpha parameter for the PhiNetConvBlock.
        :param beta: Beta parameter for the PhiNetConvBlock.
        :param t_zero: t_zero parameter for the PhiNetConvBlock.
        :param num_layers: Number of PhiNetConvBlocks in the model.
        :param latent_layer_num: Number of layers to keep as latent layers.
        :param device: Device to use for the model.
    
        """
        super().__init__()
        
        all_layers = []
        remove_ModuleList(model, all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers):
            if i < latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=10, bias=True),
        )

    def forward(self, x, latent_input=None, return_lat_acts=False):

        orig_acts=x
        if latent_input is not None:
            with torch.no_grad():
                orig_acts = self.lat_features(orig_acts)
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            with torch.no_grad():
                orig_acts = self.lat_features(orig_acts)
            lat_acts = orig_acts

        x = self.end_features(lat_acts)
        logits = self.output(x)

        if return_lat_acts:
            return logits, orig_acts
        else:
            return logits
    
if __name__ == "__main__":
    #model = PhiNet((3,32,32), alpha=0.5, beta=1, t_zero=6, num_layers=5, include_top=False)
    model = PhiNet.from_pretrained("CIFAR-10", 3.0, 0.75, 6.0, 7, 160, classifier=False)
    print("PhiNet before: ")
    print(model)
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    m = PhiNet_v3(model, latent_layer_num = 3)
    print("PhiNet after: ")
    print(m)

    """
    summary(model, (1, 28, 28))
    print("PhiNet after: ")
    print(model)
    print("--------------------------------------------------------------------------------")
    for name, param in model.named_parameters():
       print(name)
    """
