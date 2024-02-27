# Installation

* Install the packages in requirements.txt
* Cuda Runtime for `llama-cpp` with ```conda install -y -c "nvidia/label/cuda-12.1.1" cuda```
* Create a soft link to the model directory

# models


## Download
```shell
pip install huggingface-cli
```

**GGUF files**

```shell
huggingface-cli download TheBloke/Yi-34B-Chat-GGUF yi-34b-chat.Q4_K_M.gguf --local-dir models/yi-34b-chat-gguf --local-dir-use-symlinks False
```

**GPTQ files**

```shell
huggingface-cli download TheBloke/Yi-34B-Chat-GPTQ
```

## Yi

There is bug in llama.cpp in converting Yi models to `gguf`. See the fix in <https://github.com/01-ai/Yi/blob/main/docs/README_llama.cpp.md>. When using `gguf` models, one should use `chatml` prompt template. A fixed GGUF can be downloaded from onedrive.


# Notes
The AWQ model does not work properly. It never ends.