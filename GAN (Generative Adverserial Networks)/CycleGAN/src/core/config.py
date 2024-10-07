import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
report_dir = "docs/reports/model"
plot_dir = "docs/reports/data"
