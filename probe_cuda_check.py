import torch
from poc_10stocks_chronos_vs_xgboost import resolve_device, load_chronos

print('torch_version', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
print('resolved_device', resolve_device())
print('attempting Chronos load...')
pipe = load_chronos()
print('loaded_type', type(pipe).__name__)
