import torch 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
report_dir = "docs/reports/model"