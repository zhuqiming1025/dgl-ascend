import torch
import dgl
from dgl.ops import gspmm


def _check_npu_available():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return True

    try:
        import torch_npu
        return hasattr(torch, 'npu') and torch.npu.is_available()
    except ImportError:
        return False

def _synchronize_npu():
    if _check_npu_available():
        if hasattr(torch, 'npu'):
            torch.npu.synchronize()
        else:
            try:
                import torch_npu
                torch.npu.synchronize()
            except ImportError:
                pass
    


def run_gspmm(device):
    u = torch.tensor([0, 0, 1], device=device)
    v = torch.tensor([1, 2, 2], device=device)
    g = dgl.graph((u, v), device=device)
    h = torch.tensor(
        [
            [1.0, 1.0], 
            [2.0, 2.0], 
            [3.0, 3.0], 
        ],
        dtype=torch.float32,
        device=device,
    )
    e = torch.tensor(
        [
            [1.0, 0.0], 
            [0.0, 1.0], 
            [1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    out = gspmm(g, op="mul", reduce_op="sum", lhs_data=h, rhs_data=e)
    if device.type == "npu":
        _synchronize_npu()
    
    print(f"[{device}] out:")
    print(out)


if __name__ == "__main__":
    # CPU 跑一遍
    run_gspmm(torch.device("cpu"))

    # NPU（Ascend）可用则再跑一遍
    if _check_npu_available():
        run_gspmm(torch.device("npu:0"))
    else:
        print("NPU not available, skip NPU gspmm test.")