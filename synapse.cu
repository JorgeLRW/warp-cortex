#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// The Synapse is a shared memory interface between the Main Agent (Stream 1) and Side Agent (Stream 2).
// It functions as a lock-free ring buffer where the Side Agent injects "thoughts" (hidden states)
// and the Main Agent reads them via cross-attention.

#define SYNAPSE_DIM 4096  // Hidden dimension size
#define BUFFER_SIZE 128   // Number of "thought slots" in the ring buffer

struct SynapseBuffer {
    float data[BUFFER_SIZE][SYNAPSE_DIM];
    uint32_t flags[BUFFER_SIZE]; // 0 = Empty, 1 = Ready, 2 = Reading
    uint32_t head; // Written by Side Agent
    uint32_t tail; // Read by Main Agent
};

// Initialize the synapse buffer
__global__ void init_synapse(SynapseBuffer* synapse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BUFFER_SIZE) {
        synapse->flags[idx] = 0;
        for (int i = 0; i < SYNAPSE_DIM; ++i) {
            synapse->data[idx][i] = 0.0f;
        }
    }
    if (idx == 0) {
        synapse->head = 0;
        synapse->tail = 0;
    }
}

// Side Agent: Inject a thought into the synapse
// This runs on Stream 2
__global__ void inject_thought(SynapseBuffer* synapse, const float* thought_vector) {
    int tid = threadIdx.x;
    
    // Only one thread manages the head pointer, but we parallelize the copy
    __shared__ uint32_t write_idx;
    
    if (tid == 0) {
        // Atomic add to reserve a slot
        // We use a simple increment and modulo. In a real ring buffer, we'd check for overflow.
        // Here, we assume the Main Agent consumes fast enough or we overwrite (lossy memory).
        uint32_t current_head = atomicAdd(&synapse->head, 1);
        write_idx = current_head % BUFFER_SIZE;
        
        // Mark as "Writing" (optional, or just leave as 0 until done)
        synapse->flags[write_idx] = 0; 
    }
    __syncthreads();
    
    // Parallel copy of the hidden state vector
    for (int i = tid; i < SYNAPSE_DIM; i += blockDim.x) {
        synapse->data[write_idx][i] = thought_vector[i];
    }
    __syncthreads();
    
    // Mark as Ready
    if (tid == 0) {
        // Use threadfence to ensure data is visible before flag is set
        __threadfence(); 
        atomicExch(&synapse->flags[write_idx], 1);
    }
}

// Main Agent: Retrieve the latest thoughts
// This runs on Stream 1 inside the Attention mechanism
// Returns a pointer to the relevant memory slot, or NULL if no new thought
__device__ float* check_synapse(SynapseBuffer* synapse) {
    uint32_t current_tail = synapse->tail;
    uint32_t current_head = synapse->head;
    
    // If head == tail, buffer is empty
    if (current_head == current_tail) {
        return NULL;
    }
    
    uint32_t read_idx = current_tail % BUFFER_SIZE;
    
    // Check if data is ready
    if (atomicCAS(&synapse->flags[read_idx], 1, 2) == 1) {
        // Data is ready and we locked it (set to 2)
        return synapse->data[read_idx];
    }
    
    return NULL;
}

// Kernel wrapper for the Main Agent to integrate synapse data
// In reality, this would be part of the Attention Kernel
__global__ void integrate_synapse_attention(SynapseBuffer* synapse, float* query, float* output, int dim) {
    int tid = threadIdx.x;
    
    // Check for new thoughts
    float* thought = check_synapse(synapse);
    
    if (thought != NULL) {
        // Perform Cross-Attention (Simplified: Element-wise addition for demo)
        // Real implementation: Q * K_synapse
        for (int i = tid; i < dim; i += blockDim.x) {
            output[i] += thought[i] * 0.1f; // Gating factor
        }
        
        // Advance tail after reading
        if (tid == 0) {
            __threadfence();
            synapse->flags[synapse->tail % BUFFER_SIZE] = 0; // Reset flag
            atomicAdd(&synapse->tail, 1);
        }
    }
}
