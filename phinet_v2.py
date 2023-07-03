import torch
import torch.nn as nn
from micromind import PhiNet
from micromind.networks.phinet import PhiNetConvBlock, SeparableConv2d

def remove_ModuleList(network, all_layers):

    for layer in network.children():
        # if sequential layer, apply recursively to layers in sequential layer
        if isinstance(layer, nn.ModuleList):
            # print(layer)
            remove_ModuleList(layer, all_layers)
        else:  # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)


def remove_PhiNetConvBlock(cur_layers):

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

class PhiNet_v2(nn.Module):
    """MobileNet v1 implementation. This model
    can be instantiated from a pretrained network."""

    def __init__(self, pretrained : str , input_shape = (1, 28, 28), alpha = 0.5, beta = 1, t_zero = 6, num_layers = 4, latent_layer_num = 0, device = "cpu",):
        """
        :param pretrained: boolean indicating whether to load pretrained weights
        :parm latent_layer_num: determines the number of layers to consider as latent layers
        """
        super().__init__()

        # Initialize the model
        model = PhiNet(input_shape = input_shape, alpha = alpha, beta = beta, t_zero = t_zero, num_layers = num_layers).to(device)

        # Load the pretrained weights if available
        if pretrained is not None:
            unexpected_keys = ["classifier.2.weight", "classifier.2.bias"]
            state_dict = (torch.load(pretrained, map_location=torch.device(device)))
            for key in unexpected_keys:
                del state_dict[key]
            model.load_state_dict(state_dict)

        all_layers = []
        remove_ModuleList(model, all_layers)
        all_layers = remove_PhiNetConvBlock(all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=24,out_features=10, bias=True),
        )

    def forward(self, x, latent_input=None, return_lat_acts=False):

        orig_acts=x
        if latent_input is not None:
            with torch.no_grad():
                orig_acts = self.lat_features(orig_acts)
                #for layers in self.lat_features:
                #   orig_acts = layers(orig_acts)
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            #with torch.no_grad():
            orig_acts = self.lat_features(orig_acts)
            #for layers in self.lat_features:
            #        orig_acts = layers(orig_acts)
            lat_acts = orig_acts

        x = self.end_features(lat_acts)
        #x = x.view(x.size(0), -1)
        logits = self.output(x)

        if return_lat_acts:
            return logits, orig_acts
        else:
            return logits
    
if __name__ == "__main__":

    model = PhiNet_v2(latent_layer_num = 2)
    print("PhiNet:")
    print(model)

    #for name, param in model.named_parameters():
    #    print(name)
