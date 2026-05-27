from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware

import config

db = SQLAlchemy()
migrate = Migrate()

from .api.ai_router import ai_router


# ------------------------
# Flask 생성
# ------------------------

def create_flask_app():
    flask_app = Flask(__name__)
    flask_app.config.from_object(config)

    db.init_app(flask_app)
    migrate.init_app(flask_app, db)

    from . import models

    from .views import main_view
    flask_app.register_blueprint(main_view.bp)

    # flask_app.register_blueprint(question_views.bp)
    # flask_app.register_blueprint(answer_views.bp)

    return flask_app


# ------------------------
# FastAPI 생성
# ------------------------

def create_app():
    # 기존 Flask 앱 생성
    flask_app = create_flask_app()

    # FastAPI 생성
    app = FastAPI(
        title="5-Core SLM Hybrid Server",
        description="Flask(기존 웹) + FastAPI(SLM 인공지능 파이프라인) 통합 서버"
    )

    # 📌 [추가] 브라우저 UI(React, Vue, HTML 등)와 연동할 때 CORS 에러 방지
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 실제 배포시에는 UI 주소만 넣는 것을 권장합니다.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 📌 [수정] 분리된 5대 SLM 파이프라인 라우터 등록
    # /api/ai/text-gen, /api/ai/ner 등의 주소로 맵핑됩니다.
    app.include_router(ai_router, prefix="/api/ai")

    # 기존 Flask 앱을 루트("/") 경로에 마운트
    app.mount("/", WSGIMiddleware(flask_app))

    return app