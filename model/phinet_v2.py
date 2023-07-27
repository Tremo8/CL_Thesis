import torch
import torch.nn as nn
from micromind import PhiNet
from micromind.networks.phinet import PhiNetConvBlock, SeparableConv2d
from torchsummary import summary
import utility.utils as utils
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


def remove_PhiNetConvBlock(cur_layers):
    """
    Removes PhiNetConvBlock and SeparableConv2d layers from a list of layers.

    Args:
        cur_layers (list): List of layers.

    Returns:
        list: Updated list of layers with PhiNetConvBlock and SeparableConv2d layers removed.
    """
    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, (PhiNetConvBlock, SeparableConv2d)):
            for ch in layer.children():
                if isinstance(ch, nn.ModuleList):
                    ch = nn.Sequential(*ch)
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers

def get_output_shape(model):
    """
    Get the output shape of a model.

    :param model (nn.Module): The model to get the output shape of.

    :return: The output shape of the model.
    """
    device = next(model.parameters()).device 
    model.eval()
    return model(torch.rand(1, *model.input_shape).to(device)).shape[1]

class PhiNet_v2(nn.Module):
    """PhiNet implementation. This model
    can be instantiated from a pretrained network."""

    def __init__(self, pretrained : str = None, input_shape = (1, 28, 28), out_features = 10, alpha = 0.5, beta = 1, t_zero = 6, num_layers = 4, latent_layer_num = 0, replace_bn_with_brn = False, device = "cpu",):
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

        # Initialize the model
        model = PhiNet(input_shape = input_shape, alpha = alpha, beta = beta, t_zero = t_zero, num_layers = num_layers).to(device)

        # Load the pretrained weights if available
        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location=torch.device(device))
            model.load_state_dict(state_dict)
        
        if replace_bn_with_brn:
            utils.replace_bn_with_brn(model)
        all_layers = []
        remove_ModuleList(model, all_layers)
        #all_layers = remove_PhiNetConvBlock(all_layers)

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
            nn.Linear(in_features=get_output_shape(model), out_features=out_features, bias=True),
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

    model = PhiNet_v2(latent_layer_num = 3)
    print("--------------------------------------------------------------------------------")
    print("PhiNet after: ")
    print(model)
    
    
