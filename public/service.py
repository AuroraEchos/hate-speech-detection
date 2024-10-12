# Service module defining the SentimentAnalyzer and HateSpeechAnalyzer classes
# Date: 2024-10-7
# Author: Wenhao Liu

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Torch was not compiled with flash attention.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.text import tokenizer_from_json
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

import requests
import random
from hashlib import md5

class Translator:
    def __init__(self):
        self.appid = '20241012002173736'
        self.appkey = 'yqRAh4LYmE072BSLXUL8'
        self.from_lang = 'zh'
        self.to_lang = 'en'
        self.endpoint = 'http://api.fanyi.baidu.com'
        self.path = '/api/trans/vip/translate'
        self.url = self.endpoint + self.path

    @staticmethod
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    def translate(self, query):
        salt = random.randint(32768, 65536)
        sign = self.make_md5(self.appid + query + str(salt) + self.appkey)

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'appid': self.appid,
            'q': query,
            'from': self.from_lang,
            'to': self.to_lang,
            'salt': salt,
            'sign': sign
        }

        response = requests.post(self.url, params=payload, headers=headers)
        result = response.json()

        result = result['trans_result'][0]['dst']

        return result


class SentimentAnalyzer:
    """
    SentimentAnalyzer class using a pre-trained BERT model to classify emotions.
    """
    def __init__(self):
        model_path = "models\\Sentiment"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.label_map = {0: "中性", 1: "高兴", 2: "生气", 3: "伤心", 4: "恐惧", 5: "惊讶"}

    def predict(self, text):
        """
        Predicts the sentiment of the given text.
        Returns the prediction result in a structured format.
        """
        self.model.eval()
        start_time = time.time()

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predict_label = torch.argmax(logits, dim=-1).item()
            predict_confidence = torch.max(torch.softmax(logits, dim=-1)).item()

        elapsed_time = time.time() - start_time

        result = {
            'text': text,
            'predicted_emotion': self.label_map[predict_label],
            'confidence': predict_confidence,
            'elapsed_time': elapsed_time
        }

        result = results_processing(result)

        return result


class HateSpeechAnalyzer:
    """
    HateSpeechAnalyzer class using a pre-trained Keras model to detect hate speech in text.
    """
    def __init__(self):
        model_path = "models\\HateSpeech\\train_model.keras"
        tokenizer_path = "models\\HateSpeech\\tokenizer.json"
        self.max_len = 220
        self.model = load_model(model_path)
        with open(tokenizer_path, 'r') as f:
            self.tokenizer = tokenizer_from_json(f.read())
        self.threshold = 0.5
        self.auxiliary_classes = ['target', 'severe_toxicity', 'obscene', 
                                  'identity_attack', 'insult', 'threat']

    def predict(self, text):
        """
        Predicts the presence of hate speech in the given text.
        Returns the prediction results in a structured format.
        """
        start_time = time.time()
        text = [text] if isinstance(text, str) else text
        padded_sequences = pad_sequences(self.tokenizer.texts_to_sequences(text), maxlen=self.max_len)
        predictions = self.model.predict(padded_sequences, verbose=0)

        for idx, text in enumerate(text):
            main_pred_prob = predictions[0][idx][0]
            auxiliary_preds = [(self.auxiliary_classes[i], predictions[1][idx][i]) for i in range(len(self.auxiliary_classes))]

            result = {
                'text': text,
                'main_prediction_probability': float(main_pred_prob),
                'auxiliary_predictions': [
                    {'class': cls, 'probability': float(prob)} for cls, prob in auxiliary_preds  # 转换为 float
                ],
                'elapsed_time': float(time.time() - start_time)
            }

        result = results_processing(result)

        return result
    
def results_processing(results):
    """
    Translates the keys of the results from English to Chinese.

    Args:
        results (dict): The results returned from the sentiment or hate speech analysis.

    Returns:
        dict: A new dictionary with keys translated to Chinese.
    """
    translated_results = {}

    # Sentiment Analysis Results
    if 'predicted_emotion' in results:
        translated_results['文本'] = results['text']
        translated_results['预测情绪'] = results['predicted_emotion']
        translated_results['置信度'] = results['confidence']
        translated_results['耗时'] = results['elapsed_time']
    # Hate Speech Detection Results
    elif 'main_prediction_probability' in results:
        translated_results['文本'] = results['text']
        translated_results['主要预测概率'] = results['main_prediction_probability']
        translated_results['辅助预测'] = {
            '攻击目标': next((cls['probability'] for cls in results['auxiliary_predictions'] if cls['class'] == 'target'), 0),
            '高度毒性': next((cls['probability'] for cls in results['auxiliary_predictions'] if cls['class'] == 'severe_toxicity'), 0),
            '淫秽内容': next((cls['probability'] for cls in results['auxiliary_predictions'] if cls['class'] == 'obscene'), 0),
            '身份攻击': next((cls['probability'] for cls in results['auxiliary_predictions'] if cls['class'] == 'identity_attack'), 0),
            '人身侮辱': next((cls['probability'] for cls in results['auxiliary_predictions'] if cls['class'] == 'insult'), 0),
            '威胁行为': next((cls['probability'] for cls in results['auxiliary_predictions'] if cls['class'] == 'threat'), 0),
        }
        translated_results['耗时'] = results['elapsed_time']

    return translated_results

def detect_language(text):
    """
    Detects the language of the given text.
    """
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    elif re.search(r'[a-zA-Z]', text):
        return "en"
    else:
        return "unknown"


if __name__ == '__main__':

    # 情感分析
    sentiment_analyzer = SentimentAnalyzer()
    result = sentiment_analyzer.predict("我喜欢你.")
    print(result)

    # 仇恨检测
    hate_speech_analyzer = HateSpeechAnalyzer()
    result = hate_speech_analyzer.predict("Damn it, you stinky bitch, you're really disgusting.")
    print(result)