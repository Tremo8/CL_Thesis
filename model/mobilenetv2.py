import torch.nn as nn
import torch
from pytorchcv.models.common import conv1x1

def remove_sequential(network, all_layers):

    for layer in network.children():
        # if sequential layer, apply recursively to layers in sequential layer
        if isinstance(layer, nn.Sequential):
            # print(layer)
            remove_sequential(layer, all_layers)
        else:  # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)

class MobilenetV2(nn.Module):
    """MobileNet v1 implementation. This model
    can be instantiated from a pretrained network."""

    def __init__(self, model, latent_layer_num=20):
        """
        :param pretrained: boolean indicating whether to load pretrained weights
        :parm latent_layer_num: determines the number of layers to consider as latent layers
        """
        super().__init__()

        model.features.final_pool = nn.AvgPool2d(7, stride=1)

        all_layers = []
        remove_sequential(model, all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = conv1x1(
            in_channels=1280,
            out_channels=10,
            bias=False)

    def forward(self, x, latent_input=None, return_lat_acts=False):

        if latent_input is not None:
            with torch.no_grad():
                orig_acts = self.lat_features(x)
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            orig_acts = self.lat_features(x)
            lat_acts = orig_acts

        x = self.end_features(lat_acts)
        #x = x.view(x.size(0), -1)
        logits = self.output(x)
        logits = logits.view(logits.size(0), -1)

        if return_lat_acts:
            return logits, orig_acts
        else:
            return logits
        
if __name__ == "__main__":
    
    model = MobilenetV2(latent_layer_num = 15)
    print(model)