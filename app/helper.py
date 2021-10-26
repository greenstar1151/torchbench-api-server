import torch
import time
import sys

def run_one_step(func, is_cuda, niter):
    # Warm-up with one run.
    # func()

    if is_cuda:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Collect time_ns() instead of time() which does not provide better precision than 1
        # second according to https://docs.python.org/3/library/time.html#time.time.
        t0 = time.time_ns()
        func(niter)
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


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"