import time
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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

    start_time = time.time()
    # 📌 기획서 명세(946.jpg): Output text 구조 반영
    dummy_output = f"{request.text}~ 가성비 좋은 #아기유모차 #유모차 #아기용품..."
    end_time = time.time()

    print(f"📤 생성 결과: {dummy_output}")
    print("=" * 60)

    # 💥 [교정] 기획서 표준 스펙: 'output'이 아니라 'output_text' 구조로 매핑
    return {
        "status": "success",
        "pipeline": "text-gen",
        "output_text": dummy_output
    }


# --------------------------------------------------
# 2. Translation (facebook/nllb-200-distilled-600M)
# --------------------------------------------------
@ai_router.post("/translation")
async def translation(request: TextRequest, src_lang: str = "kor_Hang", tgt_lang: str = "eng_Latn"):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 2. 다국어 번역 (facebook/nllb-200)")

    start_time = time.time()
    # 📌 기획서 명세(947.jpg) 양방향 번역 아키텍처 예시 기준
    dummy_output = "Paprika automated smart farm system verification."
    end_time = time.time()

    print("=" * 60)

    return {
        "status": "success",
        "pipeline": "translation",
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "translated_text": dummy_output  # 💥 기획서 맞춤형 네이밍 교정
    }


# --------------------------------------------------
# 3. Sentiment (KoELECTRA / Sentiment)
# --------------------------------------------------
@ai_router.post("/sentiment")
async def sentiment_analysis(request: TextRequest):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 3. 감성 분석")

    start_time = time.time()
    # 📌 기획서 명세(948.jpg): 분류 라벨 및 감성 스코어(90점 등) 구조화
    label = "강한 긍정: 확신에 찬 만족도"
    score = 90.0
    end_time = time.time()

    print("=" * 60)

    return {
        "status": "success",
        "pipeline": "sentiment",
        "label": label,
        "score": score
    }


# --------------------------------------------------
# 4. NER (Davlan/bert-base-multilingual-cased-ner-hrl)
# --------------------------------------------------
@ai_router.post("/ner")
async def named_entity_recognition(request: TextRequest):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 4. 개체명 인식 (NER)")

    start_time = time.time()
    # 📌 기획서 명세(950.png): 태그명, 의미, 추출된 단어, 신뢰도 스코어 매핑
    diagnostic_table = [
        {"tag": "ORG", "meaning": "Organization (조직/기업)", "word": "삼성전자", "score": 0.9748},
        {"tag": "LOC", "meaning": "Location (지역/장소)", "word": "수원", "score": 0.9374},
        {"tag": "PER", "meaning": "Person (인물)", "word": "이재용", "score": 0.9998}
    ]
    end_time = time.time()

    print("=" * 60)

    return {
        "status": "success",
        "pipeline": "ner",
        "entities": diagnostic_table
    }


# --------------------------------------------------
# 5. QnA (monologg/koelectra-base-v3-finetuned-korquad)
# --------------------------------------------------
@ai_router.post("/qna")
async def question_answering(request: TextRequest):
    print("\n" + "=" * 60)
    print("🚀 [SLM 스레드 시작] 5. KorQuAD 질의응답 (KoELECTRA)")

    start_time = time.time()
    # 📌 기획서 명세(952.jpg): Output Box 스펙 데이터 100% 일치
    answer = "한반도 중앙에"
    score = 0.9999
    start_pos = 23
    end_pos = 30
    end_time = time.time()

    print("=" * 60)

    return {
        "status": "success",
        "pipeline": "qna",
        "answer": answer,
        "score": score,
        "start": start_pos,
        "end": end_pos
    }