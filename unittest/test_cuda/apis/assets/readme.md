# Assets for UnitTest CUDA Driver APIs

To compile out fatbin, run:

```bash
nvcc \
--generate-code arch=compute_70,code=sm_70 \
--generate-code arch=compute_72,code=sm_72 \
--generate-code arch=compute_75,code=sm_75 \
--generate-code arch=compute_80,code=sm_80 \
--generate-code arch=compute_86,code=sm_86 \
main.cu -fatbin -o sm70_72_75_80_86.fatbin
```
