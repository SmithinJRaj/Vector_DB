#include <pybind11/pybind11.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <queue>
#include <cassert>
#include <algorithm>
#include <immintrin.h>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct SearchResult {
    size_t id;
    float similarity;

    bool operator<(const SearchResult& other) const {
        return similarity >other.similarity;
    }
};

class MmapVectorDatabase {
private:
    float* data;
    size_t num_vectors;
    size_t dimensions;
    int fd;
    size_t file_size;

    std::vector<std::vector<float>> centroids;
    std::vector<std::vector<size_t>> inverted_index;

public:
    MmapVectorDatabase(const std::string& filepath, size_t dim) : dimensions(dim) {
        fd = open(filepath.c_str(), O_RDONLY);
        if(fd == -1)
        {
            perror("Failed to open vector file");
            exit(EXIT_FAILURE);
        }

        struct stat sb;
        if(fstat(fd, &sb) == -1) {
            perror("Failed to get file size");
            exit(EXIT_FAILURE);
        }
        file_size = sb.st_size;

        num_vectors = file_size / (dimensions * sizeof(float));
        std::cout << "Mapped " << num_vectors << " vectors from disk.\n";

        data = (float*)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if(data == MAP_FAILED) {
            perror("Failed to mmap file");
            exit(EXIT_FAILURE);
        }
    }

    ~MmapVectorDatabase() {
        if(munmap(data, file_size) == -1)
        {
            perror("Error unmapping memory");
        }
        close(fd);
    }

    float cosine_similarity(const float* a, const float* b) {
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        size_t i = 0;

        __m256 sum_dot = _mm256_setzero_ps();
        __m256 sum_norm_a = _mm256_setzero_ps();
        __m256 sum_norm_b = _mm256_setzero_ps();

        for(; i+7<dimensions; i += 8)
        {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);

            sum_dot = _mm256_add_ps(sum_dot, _mm256_mul_ps(va,vb));
            sum_norm_a = _mm256_add_ps(sum_norm_a, _mm256_mul_ps(va,va));
            sum_norm_b = _mm256_add_ps(sum_norm_b, _mm256_mul_ps(vb,vb));
        }

        float temp_dot[8], temp_na[8], temp_nb[8];
        _mm256_storeu_ps(temp_dot, sum_dot);
        _mm256_storeu_ps(temp_na, sum_norm_a);
        _mm256_storeu_ps(temp_nb, sum_norm_b);

        for(int j = 0; j<8; j++)
        {
            dot_product += temp_dot[j];
            norm_a += temp_na[j];
            norm_b += temp_nb[j];
        }

        for(; i<dimensions; i++)
        {
            dot_product += a[i]*b[i];
            norm_a += a[i]*a[i];
            norm_b += b[i]*b[i];
        }

        if(norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    void build_ivf_index(size_t num_clusters) {
        centroids.clear();
        inverted_index.assign(num_clusters, std::vector<size_t>());

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0,num_vectors-1);

        for(size_t i =0; i<num_clusters; i++)
        {
            size_t random_id = dist(rng);
            const float* vec_ptr = data + (random_id * dimensions);
            centroids.push_back(std::vector<float>(vec_ptr, vec_ptr + dimensions));
        }

        for (size_t i = 0; i<num_vectors; i++) {
            const float* current_vector = data + (i * dimensions);

            float best_sim = -1.0f;
            size_t best_cluster = 0;
            for (size_t c = 0; c<num_clusters; c++) {
                float sim = cosine_similarity(centroids[c].data(), current_vector);
                if(sim > best_sim) {
                    best_sim = sim;
                    best_cluster = c;
                }
            }
            inverted_index[best_cluster].push_back(i);
        }
    }

    std::vector<SearchResult> search_ivf(const std::vector<float>& query, size_t k) {
        if(centroids.empty()) throw std::runtime_error("Index not built. Call build_ivf_index first.");

        float best_sim = -1.0f;
        size_t best_cluster = 0;
        for(size_t c = 0; c < centroids.size(); c++)
        {
            float sim =cosine_similarity(query.data(), centroids[c].data());
            if(sim > best_sim)
            {
                best_sim = sim;
                best_cluster = c;
            }
        }

        std::priority_queue<SearchResult> min_heap;
        const auto& cluster_vectors = inverted_index[best_cluster];

        for(size_t vec_id : cluster_vectors)
        {
            const float* current_vector = data + (vec_id* dimensions);
            float sim = cosine_similarity(query.data(), current_vector);
            
            min_heap.push({vec_id, sim});
            if(min_heap.size() > k)
            {
                min_heap.pop();
            }
        }

        std::vector<SearchResult> results;
        while(!min_heap.empty()) {
            results.push_back(min_heap.top());
            min_heap.pop();
        }

        std::reverse(results.begin(), results.end());
        return results;
    }

    std::vector<SearchResult> search(const std::vector<float>& query, int k) {
        std::priority_queue<SearchResult> min_heap;

        for(int i=0; i<num_vectors; i++)
        {
            const float* current_vector = data + (i* dimensions);

            float sim = cosine_similarity(query.data(), current_vector);
            min_heap.push({(int)i, sim});

            if(min_heap.size() > k)
            {
                min_heap.pop();
            }
        }

        std::vector<SearchResult> results;
        while(!min_heap.empty()) {
            results.push_back(min_heap.top());
            min_heap.pop();
        }

        std::reverse(results.begin(), results.end());
        return results;
    }
};

PYBIND11_MODULE(vectordb, m) {
    m.doc() = "High-performance SIMD-accelarted Vector Database with IVF";

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("id", &SearchResult::id)
        .def_readonly("similarity", &SearchResult::similarity);
    
    py::class_<MmapVectorDatabase>(m, "VectorDatabase")
        .def(py::init<const std::string&, size_t>())
        .def("build_ivf_index", &MmapVectorDatabase::build_ivf_index)
        .def("search", &MmapVectorDatabase::search)
        .def("search_ivf", &MmapVectorDatabase::search_ivf);

}