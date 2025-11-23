
<div align="center">

# 🧪 RAG Model Evaluator
### Local SLM Verification Kit for Data Fabric

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pytorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Hardware](https://img.shields.io/badge/GPU-RTX%203090-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

<br/>

**"Is the generic SLM ready for Enterprise Data Fabric?"** <br/>
기업용 데이터 패브릭 구축을 위한 **Query Rewriter(질의 변환)** 모델 성능 검증 키트입니다.

</div>

---

## 📖 Overview

이 프로젝트는 보안이 중요한 **On-Premise(온프레미스)** 환경에서, 경량화된 **SLM(Small Language Model)** 이 복잡한 데이터베이스 검색 요청을 처리할 수 있는지 검증합니다.

- **Target**: `NEXTITS/QUANTUS-L-SLM-2509-v0.9.1`
- **Goal**: 비정형 자연어(NL)를 정형화된 메타데이터 검색 쿼리로 변환하는 능력 평가
- **Environment**: WSL2 (Ubuntu) + NVIDIA RTX 3090 (24GB)

---

## ⚡ Experimental Setup

최적의 성능과 빠른 환경 구성을 위해 **`uv`** 패키지 매니저를 사용했습니다.

| Component | Specification | Description |
| :--- | :--- | :--- |
| **OS** | 🐧 WSL2 | Ubuntu 22.04 LTS |
| **GPU** | 🟢 NVIDIA RTX 3090 | 24GB VRAM (Bfloat16 Inference) |
| **Manager** | ⚡ `uv` | High-performance Python package installer |
| **Library** | 🤗 Transformers | `accelerate`, `bitsandbytes` |

---

## 📂 Repository Structure

```bash
rag-model-tester/
├── 01_test_reasoning.py    # [Test 1] CoT(Chain of Thought) 추론 능력 검증
├── 02_test_few_shot.py     # [Test 2] Few-Shot Prompting 구조화 테스트
├── 03_test_direct.py       # [Test 3] 단순 지시(Instruction) 수행 테스트
├── requirements.txt        # 의존성 패키지 목록
└── README.md               # Result Report
```

## 🚀 Quick Start

### 1. Installation
```bash
# 1. Clone Repo
git clone https://github.com/back2zion/rag-model-tester.git
cd rag-model-tester

# 2. Setup Virtual Environment (using uv)
uv venv .venv
source .venv/bin/activate

# 3. Install Dependencies (CUDA Support)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers accelerate protobuf
```

### 2. Run Evaluation
```bash
python 01_test_reasoning.py   # Step 1: 추론 테스트
python 02_test_few_shot.py    # Step 2: 구조화 테스트
python 03_test_direct.py      # Step 3: 단순 변환 테스트
```

---

## 📊 Evaluation Results (2025.11)

**Target Model:** `NEXTITS/QUANTUS-L-SLM-2509-v0.9.1`

| Test Case | Method | Status | Findings |
| :--- | :--- | :---: | :--- |
| **1. Reasoning** | Chain of Thought | 🔴 Fail | `<think>` 토큰 미작동, 논리적 추론 실패 |
| **2. Few-Shot** | In-Context Learning | 🔴 Fail | 심각한 **환각(Hallucination)** 및 포맷 무시 |
| **3. Instruction** | Direct Prompting | 🔴 Fail | 핵심 비즈니스 키워드 매핑 실패 |

<br>

> [!IMPORTANT]
> **Conclusion: "Not Ready for Production"**
>
> 테스트 결과, 해당 모델(Base Ver.)은 **Data Fabric의 Query Rewriter로 즉시 활용하기에 부적합**합니다.
> - **문제점:** 비즈니스 도메인 용어(법인카드, 결제 등) 이해도 부족 및 지시 이행 실패.
> - **향후 계획:** 단순 프롬프트 엔지니어링이 아닌, **자체 데이터셋을 활용한 LoRA Fine-tuning** 후 재검증 예정.

<br>

<div align="center">

**Author** : 곽두일 <br>
*Data & AI Engineer / Data Fabric Researcher*

</div>
