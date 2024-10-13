# This is the main file for the Flask server. It is responsible for handling the incoming requests and sending the response back to the client.
# Date: 2024/10/14
# Author: Wehhao Liu

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from service import *


app = Flask(__name__)
app.secret_key = '021104'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    message = data['msg']
    detect_type = data.get('type') 

    try:
        lang = detect_language(message)
        if lang == 'unknown':
            response_data = {
                'type': "error",
                'input': message,
                'result': "无法识别的文本, 请检查输入"
            }
        else:
            if detect_type == 'sentiment':
                sentiment_analyzer = SentimentAnalyzer()
                result = sentiment_analyzer.predict(message)
                response_data = {
                    'type': "sentiment_analysis",
                    'input': message,
                    'result': result
                }

            elif detect_type == 'hate':
                if lang == 'zh':
                    translat = Translator()
                    message_tr = translat.translate(message)
                
                hate_speech_analyzer = HateSpeechAnalyzer()
                result = hate_speech_analyzer.predict(message_tr)
                print(result)
                response_data = {
                    'type': "hate_speech_detection",
                    'input': message,
                    'result': result
                }

    except Exception as e:
        response_data = {
            'type': "error",
            'input': message,
            'result': "服务器发生错误，请稍后再试。"
        }

    emit('response', response_data)


if __name__ == '__main__':
    print("Server is running on http://127.0.0.1:8080")
    socketio.run(app, host='127.0.0.1', port=8080)
    