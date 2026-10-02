#pragma once

#include <cstdint>

// LLAMA_PI0_PERF=1: detailed PI0 encode/decode timing on stderr.
bool llama_pi0_perf_enabled();

// PI0_DECODE_UNROLL=N (N>=2): fuse N pi05 diffusion steps per graph_compute (capped at inference_steps).
int32_t llama_pi0_decode_unroll_steps(bool is_pi05, int32_t inference_steps);

// PI0_PERSISTENT_CUDA_GRAPH=1: keep the full pi05 flow trajectory on the GPU
// and replay one stable graph for all inference steps without host round trips.
bool llama_pi0_persistent_decode_enabled();
