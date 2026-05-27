from flask import Blueprint, url_for, flash, session, request, g
from flask_sqlalchemy import SQLAlchemy
from flask import render_template
from werkzeug.utils import redirect
from werkzeug.security import generate_password_hash, check_password_hash
#from services.pipeline import run_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from sklearn.ensemble import RandomForestRegressor
from aismartfarm.forms import RegisterForm, LoginForm
from aismartfarm import db
from aismartfarm.models import Member

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('main/main.html')

#@bp.route('/main')
#def hello_flask():
#    return "hello, flask"

@bp.route('/dashboard')
def dashboard():
    return render_template('main/dashboard.html')

@bp.route('/datana')
def datana():
    return render_template('main/datana.html')

@bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file_path = 'data/파프리카가격비교.csv'

            try:
                df = pd.read_csv(file_path, encoding='cp949', header=None)
            except:
                df = pd.read_csv(file_path, encoding='euc-kr', header=None)

            first_col = df.iloc[:, 0]

            dates = df[first_col == '날짜'].iloc[0, 1:]
            volume = df[first_col == '반입량'].iloc[0, 1:]
            price_high = df[first_col == '도매가격(상)'].iloc[0, 1:]
            price_mid = df[first_col == '도매가격(중)'].iloc[0, 1:]
            price_low = df[first_col == '도매가격(하)'].iloc[0, 1:]

            clean_df = pd.DataFrame({
                'date': dates,
                'volume': volume,
                'price_high': price_high,
                'price_mid': price_mid,
                'price_low': price_low
            })

            clean_df = clean_df.replace('', np.nan)

            clean_df['volume'] = pd.to_numeric(clean_df['volume'], errors='coerce')
            clean_df['price_high'] = pd.to_numeric(clean_df['price_high'], errors='coerce')
            clean_df['price_mid'] = pd.to_numeric(clean_df['price_mid'], errors='coerce')
            clean_df['price_low'] = pd.to_numeric(clean_df['price_low'], errors='coerce')

            clean_df['date'] = pd.to_datetime(clean_df['date'], format='%Y.%m.%d', errors='coerce')

            clean_df['price'] = (
                clean_df['price_high'] * 0.5 +
                clean_df['price_mid'] * 0.3 +
                clean_df['price_low'] * 0.2
            )

            clean_df['dayofyear'] = clean_df['date'].dt.dayofyear
            clean_df['sin_day'] = np.sin(2 * np.pi * clean_df['dayofyear'] / 365)
            clean_df['cos_day'] = np.cos(2 * np.pi * clean_df['dayofyear'] / 365)

            clean_df = clean_df.dropna()

            x = clean_df[['volume', 'sin_day', 'cos_day']]
            y = clean_df['price']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(x, y)

            volume_input = float(request.form.get('volume'))
            date_input = pd.to_datetime(request.form.get('date'))

            dayofyear = date_input.dayofyear
            sin_day = np.sin(2 * np.pi * dayofyear / 365)
            cos_day = np.cos(2 * np.pi * dayofyear / 365)

            pred = model.predict([[volume_input, sin_day, cos_day]])
            result = round(pred[0], 2)

            plt.figure()

            plt.plot(clean_df['date'], clean_df['price'])
            plt.scatter(date_input, result)  # 예측값

            plt.xticks(rotation=48)

            img1 = io.BytesIO()
            plt.savefig(img1, format='png')
            img1.seek(0)

            graph_line = base64.b64encode(img1.getvalue()).decode()
            plt.close()

            plt.figure()

            plt.bar(clean_df['date'], clean_df['volume'])

            plt.xticks(rotation=48)

            img2 = io.BytesIO()
            plt.savefig(img2, format='png')
            img2.seek(0)

            graph_bar = base64.b64encode(img2.getvalue()).decode()
            plt.close()

            return render_template(
                'main/predict.html',
                result=result,
                graph_line=graph_line,
                graph_bar=graph_bar
            )

        except Exception as e:
            return f"에러 발생: {str(e)}"

    return render_template('main/predict.html')

#@bp.route('/login')
#def login():
#    return render_template('main/login.html'

@bp.route('/login/', methods=('GET', 'POST'))
def login():
        form = LoginForm()
        if request.method == 'POST' and form.validate_on_submit():
            error = None
            user = Member.query.filter_by(mem_email=form.mem_email.data).first()
            if not user:
                print("❌ 사용자 없음")
                error = "존재하지 않는 사용자입니다."
            elif not check_password_hash(user.mem_password, form.mem_password.data):
                print("❌ 비밀번호가 올바르지 않습니다.")
                error = "비밀번호가 올바르지 않습니다."

            else:
                print("✅ 로그인 성공")

            if error is None:
                session.clear()
                session['mem_email'] = user.mem_email
                session['mem_name'] = user.mem_name
                return redirect(url_for('main.index'))
            flash(error)
        return render_template('main/login.html', form=form)

@bp.before_app_request
def load_users():
    mem_name = session.get('mem_name')
    if mem_name is None:
        g.user = None
    else:
        g.user = Member.query.filter_by(mem_name=mem_name).first()

@bp.route('/logout/')
def logout():
    session.clear()
    flash("로그아웃 되었습니다.")
    return redirect(url_for('main.index'))

@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        # 🔐 이메일 중복 체크
        existing = Member.query.filter_by(mem_email=form.mem_email.data).first()
        if existing:
            return "이미 존재하는 이메일입니다."

        # 🔐 비밀번호 해싱
        hashed_pw = generate_password_hash(form.mem_password.data)

        # 🔥 DB 저장
        member = Member(
            mem_name=form.mem_name.data,
            mem_email=form.mem_email.data,
            mem_password=hashed_pw,
            mem_phone=form.mem_phone.data
        )

        db.session.add(member)
        db.session.commit()

        return redirect(url_for('main.index'))  # 메인 페이지로 이동

    return render_template('main/signup.html', form=form)

@bp.route('/ai/text-gen')
def ai_text_gen_page():
    # templates/ai/text_gen.html 화면을 브라우저에 뿌려줍니다.
    return render_template('ai/text_gen.html')

@bp.route('/ai/translation')
def ai_translation_page():
    return render_template('ai/translation.html')

@bp.route('/ai/sentiment')
def ai_sentiment_page():
    return render_template('ai/sentiment.html')

@bp.route('/ai/ner')
def ai_ner_page():
    return render_template('ai/ner.html')

@bp.route('/ai/qna')
def ai_qna_page():
    return render_template('ai/qna.html')