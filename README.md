# Tensor Parallel

A hands-on implementation of Tensor Parallelism applied to a GeLU FFN MLP.

## MLP Structure

```
x → up_proj → GeLU → down_proj → out
```

```python
B, T, H = 1, 512, 4096

up_proj   # weight: [16384, 4096]  (H → 4H)
down_proj # weight: [4096, 16384]  (4H → H)

ref = down_proj(F.gelu(up_proj(x)))  # output: [1, 512, 4096]
```

## Tensor Parallel Split (2 GPUs) — Simulation

> Simulates 2-GPU tensor parallelism on a single device. No actual inter-GPU communication is performed.

`half = 4H // 2 = 8192`

### up_proj — Column parallel (row-wise weight split)

| | weight shape | output shape |
|--|-------------|-------------|
| GPU 0 | [8192, 4096] | [1, 512, 8192] |
| GPU 1 | [8192, 4096] | [1, 512, 8192] |

```python
up_w0 = up_proj.weight[:half]  # [8192, 4096]
up_w1 = up_proj.weight[half:]  # [8192, 4096]

h0 = F.gelu(x @ up_w0.T + up_b0)  # [1, 512, 8192]
h1 = F.gelu(x @ up_w1.T + up_b1)  # [1, 512, 8192]
```

### down_proj — Row parallel (column-wise weight split)

| | weight shape | output shape |
|--|-------------|-------------|
| GPU 0 | [4096, 8192] | [1, 512, 4096] |
| GPU 1 | [4096, 8192] | [1, 512, 4096] |

```python
down_w0 = down_proj.weight[:, :half]  # [4096, 8192]
down_w1 = down_proj.weight[:, half:]  # [4096, 8192]

partial0 = h0 @ down_w0.T  # [1, 512, 4096]
partial1 = h1 @ down_w1.T  # [1, 512, 4096]
```

### AllReduce

```python
output = partial0 + partial1 + down_proj.bias  # [1, 512, 4096]
```

## Verification

```python
print(torch.allclose(output, ref, atol=1e-5))  # True
```
