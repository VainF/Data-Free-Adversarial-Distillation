from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from .mobilenet import MobileNetV2
from ._deeplab import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead
import torch.nn as nn
from torchvision.models import resnet

def _segm_mobilenet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = MobileNetV2(pretrained=pretrained_backbone)

    return_layers = {'features': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }

    inplanes = 320
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model

def deeplabv3_mobilenet(progress=True,num_classes=21, aux_loss=None, dropout_p=0.0, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _segm_mobilenet("deeplab", "mobilenet_v2", num_classes, aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model

def deeplabv3_resnet50(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_resnet("deeplab", backbone_name='resnet50', num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model