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

flash-attn의 경우 설치 후 반드시 python 내부에서 `import flash_attn`이 되는지 확인.  
Python, CUDA, Torch 버전에 따라 적합한 버전이 다를 수 있음.

```
pip install ninja "numpy>=2.0.0"
pip install "flash_attn>=2.5.8,<=2.6.3" --no-build-isolation
pip install "gradio>=4.0.0" gradio-image-prompter  # Tokenize Anything 의존성
pip install open3d spacy faiss-cpu openai imageio hydra-core distinctipy open_clip-torch "transformers>=4.41.0,<4.50.0"  # OpenGraph 의존성
```

## 서드파티 다운로드

```
git submodule update --init --recursive
```

## Recognize Anything 설치

```
pip install git+https://github.com/retia0804/recognize-anything-numpy2.git
```

## GroundingDINO 설치

thirdparties에 있는 repository는 config에 경로를 잡아 주기 위해서 추가한 내용. 패키지는 따로 설치.

```
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

## Tokenize Anything 설치

```
pip install git+https://github.com/baaivision/tokenize-anything.git
```

## SBERT 설치
```
pip install -U sentence-transformers
```

## LLAMA 설치
```
pip install git+https://github.com/meta-llama/llama.git
```

## MinkowskiEngine 설치 (4DMOS 의존성 패키지)
`MinkowskiEngineBackend` 폴더가 없으면 빌드 과정 중 실패함.
```
mkdir -p thirdparties/MinkowskiEngine/MinkowskiEngineBackend
pip install -e thirdparties/MinkowskiEngine
```

## 4DMOS 설치
```
make -C thirdparties/4DMOS install
```

# 실행

```
PYTHONPATH=$(pwd) python script/main_gen_cap.py
```