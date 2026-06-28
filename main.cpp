#include <cstdio>
#include <cstdlib>
#include <iostream>
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

struct SearchResult {
    int id;
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

void generate_dummy_data(const std::string& filepath, size_t num_vecs, size_t dim) {
    std::cout << "Generating " << num_vecs << " vectors (" << (num_vecs * dim * sizeof(float)) / (1024*1024) << " MB)...\n";
    std::ofstream out(filepath, std::ios::binary);
    
    for (size_t i = 0; i < num_vecs; ++i) {
        std::vector<float> vec(dim);
        for (size_t j = 0; j < dim; ++j) {
            // Make vector ID 4242 the perfect match, everything else random noise
            if (i == 4242) vec[j] = 1.0f; 
            else vec[j] = (float)(rand() % 100) / 100.0f;
        }
        out.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
    out.close();
    std::cout << "Binary file saved to disk.\n";
}

int main() {
    const size_t DIM = 1024;
    const std::string FILEPATH = "vectors.bin";

    // Generate 10,000 vectors (will create a ~40MB file on your disk)
    generate_dummy_data(FILEPATH, 10000, DIM);

    std::cout << "\n🚀 Booting Memory-Mapped DB...\n";
    MmapVectorDatabase db(FILEPATH, DIM);

    // Query for our target (all 1.0s)
    std::vector<float> query(DIM, 1.0f);

    std::cout << "Searching directly off the hard drive via Linux Kernel Page Cache...\n";
    auto results = db.search(query, 3);

    for (const auto& res : results) {
        std::cout << "Vector ID: " << res.id << " | Similarity: " << res.similarity << "\n";
    }

    return 0;
}