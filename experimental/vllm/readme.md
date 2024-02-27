# vLLM inference

## Installation

```shell
conda create -n vllm python=3.11
conda activate vllm
pip install vllm==0.6.4.post1
```

Note: consider move the `accelearte` installation downwards

## Results

Test with 500 images in the HICO test set. Use flash attention.

| framework   | one token (sec/img) | multiple tokens |
|:------------|---------------------|-----------------|
| vLLM        | 0.149               | 0.172           |
| transformers| 0.132               | 0.168           |
