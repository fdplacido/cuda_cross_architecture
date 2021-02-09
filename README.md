# CUDA cross compilation 

```sh
mkdir build
cmake -DSTANDALONE=ON -DTARGET_DEVICE=CUDA -DCUDA_ARCH=MAX ..
make -j4
```
