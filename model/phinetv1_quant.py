import torch
import torch.nn as nn

def get_output_shape(model):
    """
    Get the output shape of a model.

    :param model (nn.Module): The model to get the output shape of.

    :return: The output shape of the model.
    """
    device = next(model.parameters()).device 
    model.eval()
    return model(torch.rand(1, *model.input_shape).to(device)).shape[1]

class PhiNetV1(nn.Module):
    """PhiNet implementation for latent replay."""

    def __init__(self, model, latent_layer_num = 0, out_features = 10, replace_bn_with_brn = False, quantized=False):
        """
        Initialize the model.

        Args:

            model (nn.Module): The model to be used as the backbone.
            latent_layer_num (int, optional): The number of layer to be used as latent layer. Defaults to 0.
            out_features (int, optional): The number of output features. Defaults to 10.
            replace_bn_with_brn (bool, optional): Whether to replace BN with BRN. Defaults to False.
            quantized (bool, optional): Whether the feature extractor is quantized or not. Defaults to False.
    
        """
           
        super().__init__()
        self.quantized = quantized
        
        lat_list = []
        end_list = []

        for i, layer in enumerate(model._layers):
            if i < latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=model.classifier[-1].in_features, out_features=out_features, bias=True),
        )
        

    def forward(self, x, latent_input=None, return_lat_acts=False):
        orig_acts=x

        if self.quantized:
            orig_acts = orig_acts.cpu()
            latent_input = latent_input.cpu() if latent_input is not None else None
        else:
            latent_input = latent_input.to(next(self.output.parameters()).device) if latent_input is not None else None

        if latent_input is not None:
            with torch.no_grad():
                orig_acts = self.lat_features(orig_acts)
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            with torch.no_grad():
                orig_acts = self.lat_features(orig_acts)
            lat_acts = orig_acts

        if self.quantized:
            lat_acts = torch.dequantize(lat_acts)
            lat_acts = lat_acts.to(next(self.output.parameters()).device)

        x = self.end_features(lat_acts)
        logits = self.output(x)

        if return_lat_acts:
            return logits, orig_acts
        else:
            return logits