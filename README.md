### Practical Homomorphic LSTM via Programmable Bootstrapping

This code is the implementation of the paper: Practical Homomorphic LSTM via Programmable Bootstrapping.
The code compilation was done with CUDA 12.8.

### Setup
Module installation

```bash
wget https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.12.0-cuda12.tar.gz
tar -xzf nvidia-mathdx-25.12.0-cuda12.tar.gz -C third_party
rm nvidia-mathdx-25.12.0-cuda12.tar.gz
wget -q https://github.com/openxla/xla/archive/22b016fefb.tar.gz -O - | tar -xz -C third_party && mv third_party/xla-22b016fefb4cc58e200454475f20f2040d29214e third_party/xla

```


Venv + compilation
Please create a spack env "cutfhe-env" and install cuda with it.
```bash
uv sync --frozen
source .venv/bin/activate
```

### Plaintext training + Ciphertext inference
Those scripts perform training on plaintext data and ciphertext inference. Please note that each of them take more than one hour to run on NVIDIA GB200. 
```bash
python trainer_plain_infer_cipher.py --config conf_sst2.json --seed 0
python trainer_plain_infer_cipher.py --config conf_cola.json --seed 0
python trainer_plain_infer_cipher.py --config conf_mrpc.json --seed 0
```

### Comparison Plaintext/Ciphertext inference
```bash
python compare_cipher_plain.py --config conf_sst2.json
python compare_cipher_plain.py --config conf_cola.json
python compare_cipher_plain.py --config conf_mrpc.json
```

### Simulated noise 
```bash
python noise_study.py --config conf_sst2.json
python noise_study.py --config conf_cola.json
python noise_study.py --config conf_mrpc.json
```

### Speed test
In order to run this speed test, you need to have trained on a classifier on the SST-2 dataset.
```bash
python run_speed.py
```

### No Many LUT ablation
```bash
python trainer_NML.py --config conf_sst2_nml.json --seed 0
python trainer_NML.py --config conf_cola_nml.json --seed 0
python trainer_NML.py --config conf_mrpc_nml.json --seed 0
```
