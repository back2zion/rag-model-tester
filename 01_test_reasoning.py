import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 0. 환경 확인 (WSL에서 GPU가 잡히는지 체크)
print("="*50)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CPU로 실행됩니다. 속도가 느릴 수 있습니다.")
print("="*50)

# 1. 모델 로드
model_name = "NEXTITS/QUANTUS-L-SLM-2509-v0.9.1"
print(f"Loading model: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16, # RTX 3090은 bfloat16 지원 (메모리 절약)
    device_map="auto"
)

# 2. 데이터 패브릭 시나리오 (Query Rewriting)
# 상황: 현업 담당자가 구어체로 데이터를 요청 -> SQL 생성을 위한 표준 메타데이터 용어로 변환 필요
user_query = "김대리가 지난달에 법인카드 긁은 거 내역 좀 뽑아줘"

system_prompt = """
당신은 'Data Fabric Query Optimizer'입니다.
사용자의 비정형 자연어 질문을 분석하여, 데이터베이스 검색 정확도를 높이기 위해 다음 3가지 요소로 변환하여 출력하세요.
1. 핵심 키워드 추출
2. 동의어/유의어 확장
3. 정제된 검색용 문장 (Standardized Query)
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

# 3. 입력 데이터 처리
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Thinking 모드 활성화
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating response... (Thinking Process 포함)")

# 4. 추론 실행
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024, # 생각하는 과정이 길 수 있으므로 토큰 수 넉넉히
        temperature=0.1,     # 정확도 위주 설정
        do_sample=True
    )

# 5. 결과 파싱
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

try:
    # </think> 토큰(151668)의 위치를 역순으로 탐색
    reversed_ids = output_ids[::-1]
    think_end_idx = len(output_ids) - reversed_ids.index(151668)
except ValueError:
    think_end_idx = 0

thinking_content = tokenizer.decode(output_ids[:think_end_idx], skip_special_tokens=True).strip()
content = tokenizer.decode(output_ids[think_end_idx:], skip_special_tokens=True).strip()

# 6. 결과 출력
print("\n" + "="*20 + " [Thinking Process (모델의 사고 과정)] " + "="*20)
print(thinking_content)
print("\n" + "="*20 + " [Final Result (데이터 패브릭 적용 결과)] " + "="*20)
print(content)
print("="*60)