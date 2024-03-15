import torch
import sys
print(sys.version)
print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.version.cuda)
print(torch.cuda.is_available())