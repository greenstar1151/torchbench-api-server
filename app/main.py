import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException

# add path for imports inside torchbenchmark
MAIN_DIR = Path(os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(MAIN_DIR.parent.absolute(), 'pytorch_benchmark'))
from pytorch_benchmark.torchbenchmark import list_models


def run_one_step(func, cuda):
    # # Warm-up with one run.
    # func()

    if cuda:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Collect time_ns() instead of time() which does not provide better precision than 1
        # second according to https://docs.python.org/3/library/time.html#time.time.
        t0 = time.time_ns()
        func()
        t1 = time.time_ns()

        end_event.record()
        torch.cuda.synchronize()
        t2 = time.time_ns()

        # CPU Dispatch time include only the time it took to dispatch all the work to the GPU.
        # CPU Total Wall Time will include the CPU Dispatch, GPU time and device latencies.
        # print('{:<20} {:>20}'.format("GPU Time:", "%.3f milliseconds" % start_event.elapsed_time(end_event)), sep='')
        # print('{:<20} {:>20}'.format("CPU Dispatch Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')
        # print('{:<20} {:>20}'.format("CPU Total Wall Time:", "%.3f milliseconds" % ((t2 - t0) / 1_000_000)), sep='')
        return (t2 - t0) / 1_000_000, (t1 - t0) / 1_000_000, start_event.elapsed_time(end_event)

    else:
        t0 = time.time_ns()
        func()
        t1 = time.time_ns()
        #print('{:<20} {:>20}'.format("CPU Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')
        return (t1 - t0) / 1_000_000, None, None


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
            break
    if found:
        pass
        # print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode.")
    else:
        raise HTTPException(status_code=404, detail=f"Unable to find model matching '{model_name}''.")
    
    # build the model and get the chosen test method
    m = Model(device=device, jit=(mode == 'jit'))
    test = getattr(m, test)

    time_cpu_total_wall, time_cpu_dispatch, time_gpu = run_one_step(test, device == 'cuda')

    return {
        'metadata': {
            'model': Model.name,
            'device': device,
            'mode': mode,
            'task': test
        },
        'result': {
            'CPU Total Wall Time': time_cpu_total_wall, 
            'CPU Dispatch Time': time_cpu_dispatch,
            'GPU Time': time_gpu
        }
    }

