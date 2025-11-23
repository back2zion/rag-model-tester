import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*50)
print("Loading Model... (v3: Direct Instruction)")
print("="*50)

model_name = "NEXTITS/QUANTUS-L-SLM-2509-v0.9.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 상황: 복잡한 구조를 다 빼고, 아주 단순하게 "번역"처럼 시킵니다.
user_query = "김대리가 지난달에 법인카드 긁은 거 내역 좀 뽑아줘"

# 프롬프트: System Role 없이 User Role에 모든 맥락을 넣습니다.
input_text = f"""
### 지시:
아래 '입력'된 문장을 데이터베이스 검색을 위한 '정제된 검색어'로 바꿔주세요.
불필요한 설명 없이 결과만 출력하세요.

### 입력:
{user_query}

### 정제된 검색어:
"""

messages = [
    {"role": "user", "content": input_text}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1
    )

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

print("\n" + "#"*20 + " [ 최종 테스트 결과 ] " + "#"*20)
print(f"원문: {user_query}")
print("-" * 40)
print(response)
print("#"*60)