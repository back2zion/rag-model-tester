import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*50)
print("Loading Model... (v2: Few-Shot Prompting)")
print("="*50)

model_name = "NEXTITS/QUANTUS-L-SLM-2509-v0.9.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 1. 프롬프트 강화 (Data Fabric에 맞는 예시 주입)
system_prompt = """
당신은 데이터 플랫폼의 '검색 최적화 AI'입니다.
사용자의 질문을 분석하여 다음 3가지 항목으로 구조화해서 답변하세요.
반드시 아래 예시와 같은 형식을 지키세요.

[예시 1]
입력: 우리 회사 작년 매출 얼마야?
분석결과:
- 핵심키워드: 매출, 작년
- 동의어확장: Sales, Revenue, 판매금액, 전년도, 2024년
- 표준쿼리: 2024년도 전체 매출액 조회

[예시 2]
입력: 강남구 사는 우수 고객 명단 뽑아줘
분석결과:
- 핵심키워드: 강남구, 우수 고객, 명단
- 동의어확장: VIP, 서울시 강남구, 고객 리스트, 연락처, 등급
- 표준쿼리: 지역='강남구' AND 등급='VIP' 고객 정보 조회

"""

user_query = "김대리가 지난달에 법인카드 긁은 거 내역 좀 뽑아줘"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"입력: {user_query}\n분석결과:"} # 답변 시작점을 유도
]

# 2. 입력 처리 (Thinking 끄기)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # 복잡도 제거
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 3. 추론
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256, # 필요한 정보만 딱 나오게 제한
        temperature=0.1,    # 창의성 억제, 정확도 중심
        do_sample=True,
        repetition_penalty=1.2 # 같은 말 반복 방지
    )

# 4. 결과 출력
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

print("\n" + "#"*20 + " [ Data Fabric Search Result ] " + "#"*20)
print(f"사용자 질문: {user_query}")
print("-" * 40)
print(response)
print("#"*60)