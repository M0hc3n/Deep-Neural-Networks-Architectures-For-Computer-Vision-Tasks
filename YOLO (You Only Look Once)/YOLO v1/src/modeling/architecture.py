SINGLE_CONV_LAYER = "single-conv-layer"
REPEATED_CONV_LAYERS = 'repeated-conv-layers'
MAX_POOL_LAYER = "max-pool-layer"

yolo_cnn_architecture = [
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (7, 64, 2, 3)
    }, 
    {
        "type": MAX_POOL_LAYER, 
        "payload": None
    }, 
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 192, 1, 1)
    },
    {
        "type": MAX_POOL_LAYER, 
        "payload": None
    }, 
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (1, 128, 1, 0)
    }, 
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 256, 1, 1)
    }, 
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (1, 256, 1, 0)
    }, 
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 512, 1, 1)
    }, 
    {
        "type": MAX_POOL_LAYER, 
        "payload": None
    },
    {
        "type": REPEATED_CONV_LAYERS, 
        "payload": [
            (1, 256, 1, 0),
            (3, 512, 1, 1),
            4
        ]
    },  
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (1, 512, 1, 0)
    }, 
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 1024, 1, 1)
    }, 
    {
        "type": MAX_POOL_LAYER, 
        "payload": None
    },
    {
        "type": REPEATED_CONV_LAYERS, 
        "payload": [
            (1, 512, 1, 0),
            (3, 1024, 1, 1),
            2
        ]
    },  
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 1024, 1, 1)
    },
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 1024, 2, 1)
    },
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 1024, 1, 1)
    },
    {
        "type": SINGLE_CONV_LAYER, 
        "payload": (3, 1024, 1, 1)
    }
]
