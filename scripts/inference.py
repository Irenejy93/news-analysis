import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 및 토크나이저 로드
model_name = 'irene93/Llama3-news-analysis'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 모델과 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = torch.nn.DataParallel(model).to(device)

def analyze_news(user_content: str) -> str:
    """
    주어진 뉴스 콘텐츠를 분석하여 JSON 형식으로 결과를 반환합니다.
    """
    # 시스템 메시지와 사용자 메시지 설정
    messages = [
        {"role": "system", "content": "당신은 주어진 뉴스를 분석하는 챗봇입니다. **지시사항**:- 주어진 뉴스에 대하여 summary, advr, stk_code, sent_score 분석하고 json 형태로 출력하세요. - summary는 1~3줄 사이로 작성합니다.- advr는 해당 본문이 광고면 1 광고가 아닐경우에 0 으로 정수 1개의 값으로 출력하세요.- stk_code는 해당 본문에서 언급된 종목명을 찾고, 그 종목명의 종목 코드를 찾아 파이썬 리스트 형태로 작성하세요. - sent_score는 해당 본문이 긍정적일경우 1 부정적일경우 -1 , 긍정적이지도 부정적이지도 않을경우 0 으로 정수 1개의 값을 출력하세요 - 본문: 이 주어지면 결과: 다음에 json 형태로 작성하세요"},
        {"role": "user", "content": user_content}
    ]

    # 입력 데이터 생성
    input_text = f"{messages[0]['content']}\n\n본문: {messages[1]['content']}\n결과:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # 종료 토큰 설정
    terminators = [tokenizer.eos_token_id]

    # 텍스트 생성 (추론)
    with torch.no_grad():
        outputs = model.module.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators[0],
            do_sample=False,
        )

    # 출력 디코딩
    response = outputs[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(response, skip_special_tokens=True)

    return output_text

def main():
    """
    CLI를 통해 사용자 입력을 받고, 분석 결과를 출력합니다.
    """
    parser = argparse.ArgumentParser(description="LLM을 사용한 뉴스 분석")
    parser.add_argument('--input', type=str, required=True, help='분석할 뉴스 콘텐츠 (텍스트)')

    args = parser.parse_args()
    user_content = args.input

    result = analyze_news(user_content)
    print(result)

if __name__ == "__main__":
    main()
