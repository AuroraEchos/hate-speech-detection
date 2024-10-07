"""
This file defines the service module, which contains the SentimentAnalyzer class and the HateSpeechAnalyzer class.
Date: 2024-10-7
Author: Wenhao Liu
"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Torch was not compiled with flash attention.*")

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SentimentAnalyzer:
    def __init__(self):
        model_path = "models\Sentiment"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.label_map = {0: "中性", 1: "高兴", 2: "生气", 3: "伤心", 4: "恐惧", 5: "惊讶"}

    def predict(self, text, label=None):
        self.model.eval()
        
        start_time = time.time()

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predict_label = torch.argmax(logits, dim=-1).item()
            predict_confidence = torch.max(torch.softmax(logits, dim=-1)).item()

        end_time = time.time()
        elapsed_time = end_time - start_time

        info = f"inputs: '{text}', predict: '{self.label_map[predict_label]}'"
        if label is not None:
            info += f" , label: '{self.label_map[label]}'"
        
        return self.label_map[predict_label], predict_confidence, elapsed_time

def generate_emotional_reply(user_input, sentiment_analyzer):
    emotion_label, intensity, duration = sentiment_analyzer.predict(user_input)
    
    return emotion_label, intensity, duration


# 注意：这里的代码仅为示例，实际上仇恨检测模型目前并未实现，因此无法运行，仅作为演示用途
class HateSpeechAnalyzer:
    def __init__(self):
        model_path = "models/HateSpeech"  # 仇恨检测模型的路径
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        # 标签映射，0表示非仇恨言论，1表示仇恨言论
        self.label_map = {0: "非仇恨言论", 1: "仇恨言论"}

    def predict(self, text, label=None):
        self.model.eval()
        
        start_time = time.time()  # 开始计时

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predict_label = torch.argmax(logits, dim=-1).item()
            predict_confidence = torch.max(torch.softmax(logits, dim=-1)).item()

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        info = f"inputs: '{text}', predict: '{self.label_map[predict_label]}'"
        if label is not None:
            info += f" , label: '{self.label_map[label]}'"
        
        return self.label_map[predict_label], predict_confidence, elapsed_time

def generate_hate_reply(user_input, hate_speech_analyzer):
    hate_label, intensity, duration = hate_speech_analyzer.predict(user_input)
    
    return hate_label, intensity, duration



if __name__ == "__main__":
    sentiment_analyzer = SentimentAnalyzer()
    user_input = "我很生气"
    emotion_label, intensity, duration = generate_emotional_reply(user_input, sentiment_analyzer)
    print(f"情绪: {emotion_label}, 置信度: {intensity}, 耗时: {duration:.4f}秒")
