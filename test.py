import time
import vectordb

print("Booting C++ Engine from Python...")
db = vectordb.VectorDatabase("vectors.bin", 1024)
query = [1.0] * 1024

# --- 1. Exhaustive Search ---
print("\n--- Running Exhaustive Search O(N) ---")
start_time = time.time()
results_flat = db.search(query, 3)
flat_time = (time.time() - start_time) * 1000

for res in results_flat:
    print(f"Vector ID: {res.id} | Similarity: {res.similarity:.4f}")
print(f"Exhaustive Search Time: {flat_time:.2f} ms")


# --- 2. IVF Initialization ---
print("\n--- Building IVF Index (100 Clusters) ---")
start_time = time.time()
db.build_ivf_index(100) # Break the database into 100 chunks
print(f"Index built in {(time.time() - start_time) * 1000:.2f} ms")


# --- 3. IVF Search ---
print("\n--- Running IVF Search O(N/K) ---")
start_time = time.time()
results_ivf = db.search_ivf(query, 3)
ivf_time = (time.time() - start_time) * 1000

for res in results_ivf:
    print(f"Vector ID: {res.id} | Similarity: {res.similarity:.4f}")
print(f"IVF Search Time: {ivf_time:.2f} ms")

print(f"\n🚀 Speedup Factor: {flat_time / ivf_time:.2f}x faster!")