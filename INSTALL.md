# Create Environment

## Create Conda

```
conda create -n opengraph python=3.10 -y
conda activate opengraph
```

## Download Torch

> Numpy 2.x를 사용하기 위하여 torch 최소 버전
>
> - `torch>=2.3.0'
> - 'torchvision>=0.18.0`

> CUDA 버전은 11.8에서 test 되었습니다. (다른 버전에서는 안 될 가능성이 큽니다.)

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

## 의존성 설치

```
pip install ninja "numpy>=2.0.0"
pip install "gradio>=4.0.0" gradio-image-prompter  # Tokenize Anything 의존성
pip install open3d spacy faiss-cpu openai imageio hydra-core distinctipy open_clip-torch "transformers>=4.41.0,<4.50.0"  # OpenGraph 의존성
```

