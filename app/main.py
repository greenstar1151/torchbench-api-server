import os
import sys
import time
from . import helper
from .metrics import NvidiaMetrics

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
nvMetrics=NvidiaMetrics()

@app.get('/')
async def read_root():
    return {"Hello": "World"}

@app.get('/metrics')
async def metrics():
    return nvMetrics.collect_metrics()


@app.get('/list')
async def list():
    return [m.name.lower() for m in list_models()]

@app.get('/eval')
def run_model(model: str, niter: int = 1, device: str = 'cuda', mode: str = 'jit', test: str = 'eval'):
    found = False
    for Model in list_models():
        if model.lower() in Model.name.lower():
            found = True
            key=model+'_'+device
            if key not in loaded:
                loaded[key] = Model(device=device, jit=(mode == 'jit'))
            break
    if found:
        pass
        # print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode.")
    else:
        raise HTTPException(status_code=404, detail=f"Unable to find model matching '{model}'.")
    
    # build the model and get the chosen test method
    test = getattr(loaded[key], test)

    time_cpu_total_wall, time_cpu_dispatch, time_gpu = helper.run_one_step(test, device == 'cuda', niter)

    return {
        'metadata': {
            'model': Model.name,
            'device': device,
            'mode': mode,
            'task': test,
            'niter': niter,
            'size': helper.sizeof_fmt(helper.get_size(loaded[key]))
        },
        'result': {
            'CPU Total Wall Time': time_cpu_total_wall, 
            'CPU Dispatch Time': time_cpu_dispatch,
            'GPU Time': time_gpu
        }
    }

