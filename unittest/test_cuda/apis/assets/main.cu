#include <iostream>
#include <cassert>
#include <vector>

#include <cuda_runtime_api.h>

static void printMatric(std::vector<int> &matrix, int N){
    std::string row_str = "";
    for(int i=0; i<N; i++){
        row_str = "";
        for(int j=0; j<N; j++){
            row_str += std::to_string(matrix[N*i+j]);
        }
        printf("%s\n", row_str.c_str());
    }
}

static bool verifyMatrixMultiplicationResult(
    std::vector<int> &matrix_A, 
    std::vector<int> &matrix_B, 
    std::vector<int> &matrix_C,
    int N
  ){
    std::vector<int> matrix_correct;

    // precaluclate correct matrix
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            int tmp=0;
            for(int k=0; k<N; k++){
                tmp += matrix_A[N*i+k]*matrix_B[N*k+j];
            }
            matrix_correct.push_back(tmp);
        }
    }

    // for every row of matrix_C
    for(int i=0; i<N; i++){
        // for every column of matrix_C
        for(int j=0; j<N; j++){
            // assertion
            if(matrix_correct[i*N+j] != matrix_C[i*N+j]){
                printf(
                    "matmul not matched: "
                    "row_id(%d), col_id(%d), "
                    "expected(%d), obtain(%d)",
                    i, j,
                    matrix_correct[i*N+j], matrix_C[i*N+j]
                );
                printf("A:\n");
                    printMatric(matrix_A, N);
                printf("B:\n");
                    printMatric(matrix_B, N);
                printf("C:\n");
                    printMatric(matrix_C, N);
                printf("correct C:\n");
                    printMatric(matrix_correct, N);
                return false;
            }
        }
    }

    return true;
}


__global__ void squareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    const int d
){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row_index < d && col_index < d) {
    int dest_index = row_index * d + col_index;
    int temp = 0;

    for(int i = 0; i < d; i++) {
        temp += matrix_A[row_index * d + i] * matrix_B[i * d + col_index];
    }

    matrix_C[dest_index] = temp;
  }
}


int main(){
  // initialize constants
  constexpr int N = 8;
  constexpr int unit_size = sizeof(int);
  constexpr int matrix_size = N*N*unit_size;

  // create matrices (flat vectors) in host memory
  std::vector<int> matrix_A;
  matrix_A.reserve(N*N);
  std::vector<int> matrix_B;
  matrix_B.reserve(N*N);
  std::vector<int> matrix_C;
  matrix_C.reserve(N*N);

  // initialize random value for each source matrix
  for (int i=0; i<N*N; i++){
      matrix_A.push_back(rand()%100);
      matrix_B.push_back(rand()%100);
  }

  // allocate memory space on device
  int *d_matrix_A, *d_matrix_B, *d_matrix_C;
  cudaMalloc(&d_matrix_A, matrix_size);
  cudaMalloc(&d_matrix_B, matrix_size);
  cudaMalloc(&d_matrix_C, matrix_size);

  // copy data from host memory to device memory
  cudaMemcpy(d_matrix_A, matrix_A.data(), matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix_B, matrix_B.data(), matrix_size, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_matrix_C, matrix_C.data(), matrix_size, cudaMemcpyHostToDevice);

  // initialize kernel configuration
  // number of kernels per block (one dimension)
  int NUM_THREADS_PER_BLOCK = 1 << 5;

  // number of blocks in the grid (one dimension)
  int NUM_BLOCKS =  N % NUM_THREADS_PER_BLOCK == 0 ?
                    N / NUM_THREADS_PER_BLOCK :
                    N / NUM_THREADS_PER_BLOCK + 1;

  // use dim3 struct for block and grid dimensions
  dim3 threads(NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK);
  dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

  // launch kernel
  for(int i=0; i<4; i++)
    squareMatrixMul<<<blocks, threads>>>(d_matrix_A, d_matrix_B, d_matrix_C, N);
  cudaStreamSynchronize(0);

  // copy result back to host memory
  cudaMemcpy(matrix_C.data(), d_matrix_C, matrix_size, cudaMemcpyDeviceToHost);

  // free device memory
  cudaFree(d_matrix_A);
  cudaFree(d_matrix_B);
  cudaFree(d_matrix_C);

  assert(verifyMatrixMultiplicationResult(matrix_A, matrix_B, matrix_C, N));

  std::cout << "Get correct matrix multiplication result!" << std::endl;

  return 0;
}
