from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin',
               userdic='../utils/user_dic.tsv')

ner = NerModel(model_name='../models/ner/ner_model.h5', preprocess=p)
query = input()
predicts = ner.predict(query)
print(predicts)