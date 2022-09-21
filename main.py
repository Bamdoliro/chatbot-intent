import json
from fastapi import FastAPI, Response

from config.DatabaseConfig import *
from utils.Database import Database
from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel
from utils.FindAnswer import FindAnswer

app = FastAPI()

# 전처리 객체 생성
p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin',
               userdic='utils/user_dic.tsv')

# 의도 파악 모델
intent = IntentModel(model_name='models/intent/intent_model_n2.h5', preprocess=p)

# 개체명 인식 모델
ner = NerModel(model_name='models/ner/ner_model_v1.h5', preprocess=p)

db = Database(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME
)
params = {
            "db": db
        }
print("DB 접속")

@app.get("/")
def msg(msg: str):
    db = params['db']
    try:
        db.connect()  # DB 연결

        recv_json_data = json.loads(msg)
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        # 의도 파악
        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]

        # 개체명 파악
        ner_predicts = ner.predict(query)
        ner_tags = ner.predict_tags(query)

        # 답변 검색
        try:
            f = FindAnswer(db)
            answer_text, answer_image = f.search(intent_name, ner_tags)
            answer = f.tag_to_word(ner_predicts, answer_text)

        except:
            answer = "죄송해요 무슨 말인지 모르겠어요. 조금 더 공부할게요."
            answer_image = None

        return Response(content=intent_name)

    except Exception as ex:
        print(ex)

    finally:
        if db is not None:  # db 연결 끊기
            db.close()