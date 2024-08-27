from torch import nn

# a custom Layer that implements whatever lambda function passed to it
# instead of default operations (from torch)
class LambdaLayer(nn.module):
    def __init__(self, lambda_func):
        super(LambdaLayer, self).__init__()

        self.lambda_func = lambda_func

    def forward(self, x):
        return self.lambda_func(x)
