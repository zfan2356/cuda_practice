# cuda_practice
cuda best practice &amp; notes

choose the right cuda/torch version, the gpu used in this project is A10

```
micromamba install -c conda-forge cuda-toolkit=12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Run `pip install -e .` to use the kernels in the `prtc` namespace. You can then import `prtc` from the tests directory to execute and validate these kernels.