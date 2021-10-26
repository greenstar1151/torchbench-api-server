import os
import sys

from . import helper
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

# add path for imports inside torchbenchmark
MAIN_DIR = Path(os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(MAIN_DIR.parent.absolute(), 'pytorch_benchmark'))
from pytorch_benchmark.torchbenchmark import list_models

# This is used to cache already queried models in PyTorch
loaded = {}
app = FastAPI()

@app.get('/')
async def read_root():
    return {"Hello": "World"}

@app.get('/models/{model_name}')
def run_model(model_name: str, device: str = 'cuda', mode: str = 'jit', test: str = 'eval'):
    found = False
    for Model in list_models():
        if model_name.lower() in Model.name.lower():
            found = True
            if model_name not in loaded:
                loaded[model_name] = Model(device=device, jit=(mode == 'jit'))
            break
    if found:
        pass
        # print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode.")
    else:
        raise HTTPException(status_code=404, detail=f"Unable to find model matching '{model_name}''.")
    
    # build the model and get the chosen test method
    test = getattr(loaded[model_name], test)

    time_cpu_total_wall, time_cpu_dispatch, time_gpu = helper.run_one_step(test, device == 'cuda')

    return {
        'metadata': {
            'model': Model.name,
            'device': device,
            'mode': mode,
            'task': test,
            'size': helper.sizeof_fmt(helper.get_size(loaded[model_name]))
        },
        'result': {
            'CPU Total Wall Time': time_cpu_total_wall, 
            'CPU Dispatch Time': time_cpu_dispatch,
            'GPU Time': time_gpu
        }
    }

