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
    """PhiNet implementation. This model
    can be instantiated from a pretrained network."""

    def __init__(self, model, latent_layer_num = 0, out_features = 10):
        """
        Init.

        Args:
            model: PyTorch model.
            latent_layer_num: number of layers to be considered as latent.
            out_features: number of output features.    
        """
           
        super().__init__()
        
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
        """
        Forward pass.

        Args:
            x: input.
            latent_input: latent input.
            return_lat_acts: flag to return the latent activations.

        Returns:
            logits: output of the model.
            orig_acts: latent activations.
        """

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