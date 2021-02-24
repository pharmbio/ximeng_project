import torch
a = torch.zeros(3000000000, dtype=torch.int16, device='cuda:0')
b = torch.zeros(300000000, dtype=torch.int16, device='cuda:0')
# Check GPU memory using nvidia-smi
del a
del b
torch.cuda.empty_cache()
# Check GPU memory again
torch.cuda.memory_cached()
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()

torch.backends.cuda.cufft_plan_cache.size
torch.backends.cuda.cufft_plan_cache.clear()
import torch
print(torch.__version__) 
#python /home/jovyan/repo/ximeng_project/split0.2_pytorch_ResNet50_nparray.py &> /home/jovyan/repo/ximeng_project/Outputs/TerminalOutput0219resnet50.txt
#python /home/jovyan/repo/ximeng_project/split0.2_ResNet50_freeze.py &> /home/jovyan/repo/ximeng_project/Outputs/TerminalOutput0222resnet50freeze.txt
#python /home/jovyan/repo/ximeng_project/split0.2_pytorch_VGG19.py &> /home/jovyan/repo/ximeng_project/Outputs/TerminalOutput0222vgg19.txt

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
model = AutoModelWithLMHead.from_pretrained("t5-large") # same behavior for `bert-base-uncased`, larger T5 models..
model = model.torch.cuda.device("cuda:1")
model = model.train()

## delete model 
del model 
torch._C._cuda_emptyCache()
## alternatively 
# with torch.cuda.device("cuda:0"): 
#   ...:     torch.cuda.empty_cache()

## (as per the discussion here: https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3, seeing all the hanging tensors)
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass