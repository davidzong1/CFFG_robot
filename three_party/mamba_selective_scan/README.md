# mamba-selective-scan

从 [Mamba](https://github.com/state-spaces/mamba) 官方仓库提取的 **Selective Scan** 高性能 CUDA 实现。

## 快速安装

```bash
pip install git+https://github.com/biubushy/mamba-selective-scan.git
```

## 来源说明

本包**仅做提取工作**，所有 CUDA 算子代码原封不动地来自 Mamba 官方仓库的指定提交：

- **仓库**: https://github.com/state-spaces/mamba
- **Commit**: [`f1493ff6e9335160eb134eb67e59f8e4d9adefd6`](https://github.com/state-spaces/mamba/tree/f1493ff6e9335160eb134eb67e59f8e4d9adefd6)
- **提取范围**: `csrc/selective_scan/` 目录下的全部 CUDA/C++ 源文件及对应的 Python 封装层

**本包未对原始算法进行任何修改**，仅将 Selective Scan 这一子模块从完整的 Mamba 项目中独立提取出来，使其可作为轻量级依赖单独安装。

原始算法作者: **Albert Gu**, **Tri Dao**

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.6 (NVIDIA) 或 ROCm >= 6.0 (AMD)

## API

### `selective_scan_fn`

CUDA 加速的 Selective Scan 前向 + 反向（通过 `torch.autograd` 自动支持）。

```python
from mamba_selective_scan import selective_scan_fn

out = selective_scan_fn(
    u,              # (batch, dim, seqlen)
    delta,          # (batch, dim, seqlen)
    A,              # (dim, dstate)
    B,              # (dim, dstate) 或 (batch, ngroups, dstate, seqlen)
    C,              # (dim, dstate) 或 (batch, ngroups, dstate, seqlen)
    D=None,         # (dim,)
    z=None,         # (batch, dim, seqlen) — 门控
    delta_bias=None,        # (dim,)
    delta_softplus=False,
    return_last_state=False,
)
```

### `selective_scan_ref`

纯 Python/PyTorch 参考实现，不依赖 CUDA 扩展，可用于调试和正确性验证。函数签名与 `selective_scan_fn` 一致。

```python
from mamba_selective_scan import selective_scan_ref
```

## 许可证

本包遵循 [Apache License 2.0](LICENSE)，与 Mamba 官方仓库保持一致。
