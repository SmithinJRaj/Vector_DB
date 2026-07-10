# ApexVec 🚀

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![SIMD](https://img.shields.io/badge/AVX2-SIMD-red)
![License](https://img.shields.io/badge/License-MIT-green)

> A high-performance C++17 vector search engine featuring AVX2 SIMD acceleration, memory-mapped storage, and Approximate Nearest Neighbor (ANN) search through an Inverted File (IVF) index.

ApexVec is a lightweight vector search engine written in modern **C++17** and exposed to Python using **pybind11**. It is designed to efficiently search high-dimensional embedding vectors while minimizing memory overhead through POSIX `mmap()` and accelerating similarity computations with Intel AVX2 SIMD instructions.

---

# Why ApexVec?

Modern AI applications rely heavily on vector similarity search for tasks such as semantic search, recommendation systems, Retrieval-Augmented Generation (RAG), and image retrieval.

ApexVec focuses on the core search engine itself rather than distributed infrastructure. It demonstrates how low-level systems programming techniques can significantly improve vector search performance by combining:

- AVX2 SIMD instructions
- Memory-mapped file access (`mmap`)
- Approximate Nearest Neighbor (IVF)
- Native Python bindings

The project serves both as a learning resource for systems programming and as a lightweight local vector search engine.

---

# ✨ Features

- ⚡ **AVX2 SIMD Acceleration** — Computes cosine similarity using Intel AVX2 intrinsics.
- 💾 **Memory-Mapped Storage** — Uses POSIX `mmap()` for efficient demand-paged access to vector files.
- 🔍 **Approximate Nearest Neighbor Search** — Simplified IVF index for faster search than exhaustive scanning.
- 🐍 **Native Python Module** — Exposed through pybind11 with zero wrapper code required.
- 📈 **Top-K Search** — Maintains only the best K results using a bounded priority queue.
- 🖥 **Cross Platform** — Supports Linux and macOS.

---

# 🏗 Architecture

```text
                Python Application
                        │
                        │ pybind11
                        ▼
             +----------------------+
             |     ApexVec Engine   |
             +----------------------+
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 AVX2 Cosine Similarity          IVF Index
        │                     (Random Centroids)
        └───────────────┬───────────────┘
                        ▼
              POSIX mmap() Storage
                        │
                        ▼
             vectors.bin (float32 data)
```

---

# 🧠 How It Works

## 1. Memory-Mapped Storage

Instead of reading every vector into heap memory, ApexVec maps the binary vector file directly into the process's virtual address space using POSIX `mmap()`.

Advantages:

- Lower startup memory usage
- Efficient access to large datasets
- OS-managed page caching
- Eliminates explicit file buffering

---

## 2. SIMD Cosine Similarity

Computing cosine similarity is the most expensive part of vector search.

Rather than processing one floating-point value at a time, ApexVec uses AVX2 SIMD instructions to process **8 single-precision floating-point values simultaneously**.

```
[a0 a1 a2 a3 a4 a5 a6 a7]
            ×
[b0 b1 b2 b3 b4 b5 b6 b7]
```

The partial sums are accumulated in SIMD registers before being reduced into the final cosine similarity score.

---

## 3. Inverted File (IVF) Index

Searching every vector becomes increasingly expensive as datasets grow.

ApexVec implements a simplified IVF index:

1. Randomly selects vectors as centroids.
2. Assigns each database vector to its nearest centroid.
3. During search:
   - Finds the nearest centroid.
   - Searches only vectors belonging to that cluster.

This significantly reduces the number of vectors evaluated during search while maintaining good retrieval quality.

> **Note:** This implementation uses randomly selected centroids rather than iterative K-Means training.

---

## 4. Top-K Retrieval

Search results are maintained using a bounded priority queue.

```
Priority Queue (size = K)

0.99
0.97
0.95
────────────
Discard lower similarities
```

Memory usage remains **O(K)** regardless of dataset size.

---

# 📂 Project Structure

```
.
├── engine.cpp
├── vectors.bin
├── README.md
└── LICENSE
```

---

# ⚙ Build

## Requirements

- Linux or macOS
- C++17 Compiler
- Python 3.x
- pybind11

Install pybind11:

```bash
pip install pybind11
```

Compile:

```bash
c++ -O3 \
-Wall \
-shared \
-std=c++17 \
-fPIC \
$(python3 -m pybind11 --includes) \
engine.cpp \
-o vectordb$(python3-config --extension-suffix) \
-march=native
```

The `-march=native` flag enables CPU-specific optimizations such as AVX2 when supported by the host processor.

---

# 🚀 Quick Start

## Generate Example Data

```python
import random
import struct

DIMENSIONS = 1024
NUM_VECTORS = 10000

with open("vectors.bin", "wb") as f:
    for _ in range(NUM_VECTORS):
        vector = [random.random() for _ in range(DIMENSIONS)]
        f.write(struct.pack(f"{DIMENSIONS}f", *vector))
```

---

## Search

```python
import time
import vectordb

DIMENSIONS = 1024

db = vectordb.VectorDatabase("vectors.bin", DIMENSIONS)

print("Building IVF index...")
db.build_ivf_index(100)

query = [1.0] * DIMENSIONS

start = time.time()

results = db.search_ivf(query, 3)

end = time.time()

print("Top Results")

for result in results:
    print(
        f"Vector ID: {result.id} | Similarity: {result.similarity:.4f}"
    )

print(f"Search Time: {(end-start)*1000:.3f} ms")
```

---

# 📚 API

## Constructor

```python
VectorDatabase(filepath, dimensions)
```

Creates a memory-mapped vector database.

---

## Build IVF Index

```python
build_ivf_index(num_clusters)
```

Builds the approximate nearest neighbor index.

---

## Exact Search

```python
search(query, k)
```

Performs exhaustive cosine similarity search over all vectors.

Returns:

```python
[
    SearchResult(id=..., similarity=...)
]
```

---

## Approximate Search

```python
search_ivf(query, k)
```

Searches only the nearest IVF cluster for faster approximate retrieval.

---

# 📊 Performance

Benchmark performed on:

- 10,000 vectors
- 1024 dimensions
- ~40 MB binary dataset

| Search Method | Query Latency |
|---------------|--------------:|
| Exact Search | ~3.53 ms |
| IVF Search | ~0.56 ms |

Observed speedup:

```
≈ 6× faster
```

Performance depends on:

- CPU architecture
- SIMD support
- Storage device
- Number of clusters
- Dataset size

---

# 🚧 Current Limitations

- Simplified IVF implementation using randomly selected centroids.
- Read-only database.
- IVF index is rebuilt at startup.
- No index serialization.
- No incremental vector insertion or deletion.
- CPU-only implementation.

---

# 🚀 Future Improvements

- Iterative K-Means centroid training
- Multi-probe IVF search
- OpenMP parallel search
- HNSW support
- Product Quantization (PQ)
- Index serialization
- ARM NEON support for Apple Silicon
- AVX-512 optimization

---

# 🛠 Technologies

- C++17
- Intel AVX2 Intrinsics
- POSIX mmap
- pybind11
- STL
- Priority Queue
- Python

---

# 📄 License

This project is licensed under the **MIT License**.