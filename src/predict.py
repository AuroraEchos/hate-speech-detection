# Description: This script defines a class that can be used to load a trained toxicity classification model and make predictions on new text data.
# Date: 2024-10-12
# Name: Wenhao Liu

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.text import tokenizer_from_json
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from prettytable import PrettyTable

class ToxicityPredictor:
    def __init__(self, model_path, tokenizer_path, max_len=220):

        # Load the trained model
        self.model = load_model(model_path)

        # Load the tokenizer
        with open(tokenizer_path, 'r') as f:
            self.tokenizer = tokenizer_from_json(f.read())

        self.max_len = max_len
        self.threshold = 0.5
        self.auxiliary_classes = ['target', 'severe_toxicity', 'obscene', 
                                  'identity_attack', 'insult', 'threat']

    def predict(self, texts):
        # Preprocess the text (tokenize and pad)
        padded_sequences = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.max_len)

        # Run inference
        predictions = self.model.predict(padded_sequences)

        results = []
        for idx, text in enumerate(texts):
            main_pred_prob = predictions[0][idx][0]

            auxiliary_preds = [
                (self.auxiliary_classes[i], predictions[1][idx][i]) 
                for i in range(len(self.auxiliary_classes))
            ]

            results.append({
                'text': text,
                'main_prediction_probability': main_pred_prob,
                'auxiliary_predictions': [
                    {
                        'class': cls,
                        'probability': prob
                    } for cls, prob in auxiliary_preds
                ]
            })
        return results

    def pretty_print(self, predictions):
        table = PrettyTable()
        table.field_names = ["Text", "Main Prediction Probability"] + self.auxiliary_classes

        for result in predictions:
            row = [result['text'], f"{result['main_prediction_probability']:.4f}"]
            row += [f"{aux['probability']:.4f}" for aux in result['auxiliary_predictions']]
            table.add_row(row)

        print(table)

# Example
if __name__ == "__main__":
    predictor = ToxicityPredictor('model/train_model.keras', 'model/tokenizer.json')

    texts = [
        "You ignorant fools are really disgusting.",
        "I hate everyone who disagrees with me. Stinky bitch."
    ]

    predictions = predictor.predict(texts)
    predictor.pretty_print(predictions)


'''
输出结果说明

主要预测结果 (Main Prediction):
- Main Prediction (Probability): 表示文本被分类为“有毒”或“无毒”的概率。接近1的值（如0.85）表示文本可能被认为是有毒的。
- Main Prediction (Class): 根据设定的阈值（通常为0.5）进行的二元分类。值为1表示文本被分类为有毒，0表示无毒。

辅助预测结果 (Auxiliary Predictions):
- 辅助输出提供了更细致的分类结果，包括：
  - target: 指示文本是否为仇恨语言。
  - severe_toxicity: 是否包含严重的仇恨或毒性语言。
  - obscene: 是否包含粗俗语言。
  - identity_attack: 是否针对特定群体或个体的攻击。
  - insult: 是否包含侮辱性语言。
  - threat: 是否包含威胁或暴力言论。
'''


