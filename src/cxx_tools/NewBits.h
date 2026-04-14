#include <cstdint>
#include <cstddef>
#include <omp.h>


void pack_bits_omp(const uint8_t *input, uint32_t *output, size_t src_size);

void unpack_bits_omp(const uint32_t *input, uint8_t *output, size_t dest_size);