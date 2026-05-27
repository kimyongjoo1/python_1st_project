import time
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

ai_router = APIRouter()

class TextRequest(BaseModel):
    text: str
    context: Optional[str] = None


# --------------------------------------------------
# 1. Text Gen (sk-t/kogpt2-base-v2)
# --------------------------------------------------
@ai_router.post("/text-gen")
async def text_generation(request: TextRequest):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 1. 텍스트 생성 (skt/kogpt2-base-v2)")
    print(f"📥 수신된 본문 텍스트: {request.text}")

    start_time = time.time()
    # TODO: 여기에 실제 모델 추론 코드 삽입 (예: result = text_generator(...))
    dummy_output = f"{request.text} ~ 스마트팜 환경에서 가성비와 당도가 우수한 파프리카 재배 조절 장치 특허 출원."
    end_time = time.time()

    print(f"⏱️ 모델 추론 소요 시간: {end_time - start_time:.4f}초")
    print(f"📤 생성 결과: {dummy_output[:40]}...")
    print("=" * 60)

    return {"status": "success", "pipeline": "text-gen", "output": dummy_output}


# --------------------------------------------------
# 2. Translation (facebook/nllb-200)
# --------------------------------------------------
@ai_router.post("/translation")
async def translation(request: TextRequest, src_lang: str = "kor_Hang", tgt_lang: str = "eng_Latn"):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 2. 다국어 번역 (facebook/nllb-200)")
    print(f"📥 번역 대상 원문: {request.text}")

    start_time = time.time()
    dummy_output = "Paprika automated smart farm system verification."
    end_time = time.time()

    print(f"⏱️ 번역 모델 추론 소요 시간: {end_time - start_time:.4f}초")
    print(f"📤 영문 변환 완료: {dummy_output}")
    print("=" * 60)

    return {"status": "success", "pipeline": "translation", "src_lang": src_lang, "tgt_lang": tgt_lang,
            "output": dummy_output}


# --------------------------------------------------
# 3. Sentiment (KoELECTRA / Sentiment)
# --------------------------------------------------
@ai_router.post("/sentiment")
async def sentiment_analysis(request: TextRequest):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 3. 감성 분석")
    print(f"📥 분석 대상 문장: {request.text}")

    start_time = time.time()
    label = "강한 긍정 (스마트팜 도입 만족)"
    score = 94.85
    end_time = time.time()

    print(f"⏱️ 감성 분석 소요 시간: {end_time - start_time:.4f}초")
    print(f"📤 판정: {label} ({score}%)")
    print("=" * 60)

    return {"status": "success", "pipeline": "sentiment", "label": label, "score": score}


# --------------------------------------------------
# 4. NER (Davlan/bert-base-multilingual-cased-ner-hrl)
# --------------------------------------------------
@ai_router.post("/ner")
async def named_entity_recognition(request: TextRequest):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 4. 개체명 인식 (NER)")
    print(f"📥 데이터 비정형 텍스트: {request.text}")

    start_time = time.time()
    diagnostic_table = [
        {"tag": "ORG", "meaning": "Organization", "word": "농림축산식품부", "score": 0.9855},
        {"tag": "LOC", "meaning": "Location", "word": "서울 양재동", "score": 0.9412},
        {"tag": "PER", "meaning": "Person", "word": "김용주", "score": 0.9991}
    ]
    end_time = time.time()

    print(f"⏱️ NER 개체 추출 시간: {end_time - start_time:.4f}초")
    print(f"📤 검출된 개체 수: {len(diagnostic_table)}개")
    print("=" * 60)

    return {"status": "success", "pipeline": "ner", "entities": diagnostic_table}


# --------------------------------------------------
# 5. QnA (monologg/koelectra-base-v3-finetuned-korquad)
# --------------------------------------------------
@ai_router.post("/qna")
async def question_answering(request: TextRequest):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 5. KorQuAD 질의응답 (KoELECTRA)")
    print(f"📥 본문(Context): {request.context}")
    print(f"📥 사용자 질문(Question): 파프리카 적정 재배 온도는?")

    start_time = time.time()
    answer = "섭씨 20도에서 22도 사이"
    score = 0.9987
    end_time = time.time()

    print(f"⏱️ 지식 답변 추출 시간: {end_time - start_time:.4f}초")
    print(f"📤 최종 기계 독해 정답: {answer}")
    print("=" * 60)

    return {"status": "success", "pipeline": "qna", "answer": answer, "score": score, "start": 14, "end": 28}