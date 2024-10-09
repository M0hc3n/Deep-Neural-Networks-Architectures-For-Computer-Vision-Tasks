import torch
import torch.nn as nn
import torch.optim as optim

from core.config import device

from modeling.architecture import yolo_cnn_architecture, REPEATED_CONV_LAYERS, MAX_POOL_LAYER, SINGLE_CONV_LAYER
from modeling.cnn_block import CNNBlock

class Yolo(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolo, self).__init__()
        
        self.in_channels = in_channels
        self.net = self._build_layers()
        
        # reset the in_channels value
        # self.in_channels = in_channels
        
        self.fcs = self._build_fcs(**kwargs)
        
    def _get_conv_layer(self, layer):
        res = [
            CNNBlock(
            in_channels=self.in_channels, 
            out_channels=layer[1], 
            kernel_size=layer[0],
            stride=layer[2], 
            padding=layer[3]
        )]
        
        self.in_channels = layer[1]
        
        return res
        
    
    def _get_maxpool_layer(self):
        return [
            nn.MaxPool2d(
                kernel_size=(2, 2), 
                stride=(2, 2), 
            )
        ]

    def _get_repeated_conv_layers(self, layers):
        layer_1 = layers[0]
        layer_2 = layers[1]
        
        repititions = layers[2]
        
        res = []
        
        for _ in range(repititions):
            res += self._get_conv_layer(layer_1)
            res += self._get_conv_layer(layer_2)
        
        return res
    
    def _get_layer_by_type(self, layer):
        layer_type = layer["type"]
        
        if layer_type == SINGLE_CONV_LAYER:
            return self._get_conv_layer(layer["payload"])
        elif layer_type == MAX_POOL_LAYER:
            return self._get_maxpool_layer()
        elif layer_type == REPEATED_CONV_LAYERS:
            return self._get_repeated_conv_layers(layer["payload"])
        else:
            raise Exception("This Layer Type Has No Corresponding Implementation")
        
    def _build_layers(self):
        layers = []
        
        for _layer in yolo_cnn_architecture:
            layers += self._get_layer_by_type(_layer)
            
        return nn.Sequential(*layers)
    
    def _build_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        
        return nn.Sequential(
            nn.Flatten(), 
            nn.Linear(1024 * S * S, 496), 
            nn.Dropout(0.0), 
            nn.LeakyReLU(0.1), 
            nn.Linear(496, S * S * (C + B * 5))
        )
        
    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)
    
        
        
        
        
        

      