import torch
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.resnet import ResNet152_Weights
from torchvision.models.swin_transformer import Swin_T_Weights
from torchvision.models.swin_transformer import Swin_V2_T_Weights
from torchvision.models.inception import Inception_V3_Weights
from monai.networks.nets import EfficientNetBN

from monai.networks.nets import TorchVisionFCModel
import sys

def classification_model(
    model_name: str = 'resnet50',
    num_classes: int = 6,
):
    if model_name == 'resnet50':
        model =  TorchVisionFCModel(model_name='resnet50', 
                     num_classes=num_classes, 
                     dim=2, 
                     in_channels=None,
                     use_conv=False, 
                     pool=None,
                     bias=True,
                     weights=ResNet50_Weights.IMAGENET1K_V2
                )
    
    elif model_name == 'resnet152':
        model = TorchVisionFCModel(model_name='resnet152', 
                           num_classes=num_classes, 
                           dim=2, 
                           in_channels=None, 
                           use_conv=False, 
                           pool=None, 
                           bias=True,
                           weights=ResNet152_Weights.IMAGENET1K_V2
                )
        
    elif model_name == 'inception_v3':
        model = TorchVisionFCModel(model_name='inception_v3', 
                           num_classes=num_classes, 
                           dim=2, 
                           in_channels=None, 
                           use_conv=False, 
                           pool=None,
                           bias=True,
                           weights=Inception_V3_Weights.IMAGENET1K_V1
                )

    elif model_name == 'swin_t':
        model = TorchVisionFCModel(model_name='swin_t', 
                           num_classes=num_classes, 
                           in_channels=None, 
                           use_conv=False, 
                           pool=None,
                           bias=True,
                           weights=Swin_T_Weights.IMAGENET1K_V1
                )
        
    elif model_name == 'swin_v2_t':
        model = TorchVisionFCModel(model_name='swin_v2_t', 
                           num_classes=num_classes, 
                           in_channels=None, 
                           use_conv=False, 
                           pool=None, 
                           bias=True,
                           weights=Swin_V2_T_Weights.IMAGENET1K_V1
                )

    elif model_name == 'efficientnet-b7':
        model = EfficientNetBN( 
            "efficientnet-b7",
            spatial_dims=2,            
            in_channels=3,
            num_classes=num_classes,
            pretrained = True
        )
    
    else:
        print('Model requested is not recognized')
        sys.exit(0)
    
    return model