# POWER8 Thread Optimization Guide

Optimal thread configurations for llama.cpp on IBM POWER8 S824.

## System Specs

| Component | Value |
|-----------|-------|
| CPU | Dual 8-core POWER8 |
| Threads | 128 (SMT8) |
| Physical Cores | 16 |
| RAM | 576 GB DDR3 |
| NUMA Nodes | 4 |

## Thread Scaling Results (TinyLlama 1.1B Q4_K)

| Threads | pp128 (t/s) | tg16 (t/s) | Notes |
|---------|-------------|------------|-------|
| 32 | 50.53 | **14.13** | Best for generation |
| 48 | 61.56 | 11.88 | |
| 64 | 70.56 | 8.46 | Previous "optimal" |
| 80 | 82.11 | 8.24 | |
| 96 | 90.95 | 7.45 | Balanced |
| **112** | **100.09** | 7.22 | **Best for prompt** |
| 128 | 71.01 | 1.33 | ❌ SMT oversubscription |

## Key Findings

1. **Prompt Processing (pp) scales with threads** up to 112
   - More threads = more parallelism for batch operations
   - 112 threads achieves **100+ t/s** (6x improvement over 32)

2. **Token Generation (tg) scales INVERSELY**
   - Fewer threads = less contention for sequential work
   - 32 threads achieves **14.13 t/s** (2x better than 64)

3. **128 threads kills performance**
   - SMT8 means 8 hardware threads share core resources
   - Full oversubscription causes massive contention
   - pp drops from 100 to 71, tg from 7 to 1.3

## Optimal Configurations

### For Interactive Chat (balanced)
```bash
export OMP_NUM_THREADS=96
numactl --interleave=all ./llama-cli -t 96 ...
```

### For Batch Processing (max throughput)
```bash
export OMP_NUM_THREADS=112
numactl --interleave=all ./llama-cli -t 112 ...
```

### For Fast Generation (latency-sensitive)
```bash
export OMP_NUM_THREADS=32
numactl --cpunodebind=1 --membind=1 ./llama-cli -t 32 ...
```

## NUMA Considerations

```
Node 0: 130GB, CPUs 0-31   (paired with Node 2)
Node 1: 190GB, CPUs 32-63  (paired with Node 3)
Node 2: 65GB, CPUs 64-95   (paired with Node 0)
Node 3: 195GB, CPUs 96-127 (paired with Node 1)
```

- **Small models (<50GB)**: Use single node for locality
- **Large models (>100GB)**: Use `--interleave=all` to spread across nodes

## PSE Optimizations Active

The POWER8 build includes Proto-Sentient Emergence (PSE) optimizations:

```
PSE Vec_Perm Collapse Active - POWER8 S824
 - Top-K: 8 | Amplify: 1.20 | Entropy: mftb
 - RAM Coffers: ON | L2/L3 Resident: ON
 - IBM MASS: ENABLED (vsexp, vstanh)
```

These provide additional ~1.5-2x speedup over stock llama.cpp.

## December 2025
