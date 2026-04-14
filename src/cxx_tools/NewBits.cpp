#include "NewBits.h"

void pack_bits_omp(const uint8_t *input, uint32_t *output, size_t src_size) {
    size_t num_words = src_size >> 5; // src / 32

    #pragma omp parallel for schedule(static) num_threads(32)
    for (size_t i = 0; i < num_words; ++i) {
        uint32_t word = 0;
        const uint8_t* chunk = &input[i << 5];

        // Inner loop builds the word in a local register
        for (int bit = 0; bit < 32; ++bit) {
            if (chunk[bit]) {
                word |= (1U << bit);
            }
        }
        output[i] = word;
    }
}

void unpack_bits_omp(const uint32_t* src, uint8_t* dest, size_t dest_size) {
    size_t num_words = dest_size >> 5; // dest_size / 32

    // Parallelize the expansion. Each thread handles a subset of the source words.
    #pragma omp parallel for schedule(static) num_threads(32)
    for (size_t i = 0; i < num_words; ++i) {
        uint32_t word = src[i];
        uint8_t* chunk = &dest[i << 5]; // Offset into dest by i * 32

        // Extract 32 bits from the single word and write them to 32 integers
        for (int bit = 0; bit < 32; ++bit) {
            // Check if the bit at the current offset is set
            // The !! (double negation) or a simple comparison ensures we store exactly 1 or 0
            chunk[bit] = (word >> bit) & 1U;
        }
    }
}