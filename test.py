import time
import vectordb

DIMENSIONS = 1024
TOP_K = 3
NUM_CLUSTERS = 100

print("🚀 Booting ApexVec...")
db = vectordb.VectorDatabase("vectors.bin", DIMENSIONS)

query = [1.0] * DIMENSIONS

# -------------------------------------------------
# Exact Search
# -------------------------------------------------
print("\n=== Exact Search ===")

start = time.perf_counter()
results_flat = db.search(query, TOP_K)
flat_time = (time.perf_counter() - start) * 1000

for result in results_flat:
    print(
        f"Vector ID: {result.id} | Similarity: {result.similarity:.4f}"
    )

print(f"Exact Search Time: {flat_time:.3f} ms")

# -------------------------------------------------
# Build IVF Index
# -------------------------------------------------
print("\n=== Building IVF Index ===")

start = time.perf_counter()
db.build_ivf_index(NUM_CLUSTERS)
build_time = (time.perf_counter() - start) * 1000

print(f"Index Build Time: {build_time:.3f} ms")

# -------------------------------------------------
# IVF Search
# -------------------------------------------------
print("\n=== IVF Search ===")

start = time.perf_counter()
results_ivf = db.search_ivf(query, TOP_K)
ivf_time = (time.perf_counter() - start) * 1000

for result in results_ivf:
    print(
        f"Vector ID: {result.id} | Similarity: {result.similarity:.4f}"
    )

print(f"IVF Search Time: {ivf_time:.3f} ms")

if ivf_time > 0:
    print(f"\n🚀 Speedup: {flat_time / ivf_time:.2f}× faster")