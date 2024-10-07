from flask import Flask, render_template, request, session
from flask_socketio import SocketIO, emit
from gevent import pywsgi

from service import *


app = Flask(__name__)
app.secret_key = '021104'
socketio = SocketIO(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    message = data['msg']
    detect_type = data.get('type') 

    try:
        if detect_type == 'sentiment':
            sentiment_analyzer = SentimentAnalyzer()
            emotion_label, intensity, duration = sentiment_analyzer.predict(message)

            response_data = {
                'input': message,
                'result': f"情绪: {emotion_label}, 置信度: {intensity:.4f}, 耗时: {duration:.4f}秒"
            }

        elif detect_type == 'hate':
            """ hate_speech_analyzer = HateSpeechAnalyzer()
            hate_label, intensity, duration = hate_speech_analyzer.predict(message)

            response_data = {
                'input': message,
                'result': f"检测结果: {hate_label}, 置信度: {intensity:.4f}, 耗时: {duration:.4f}秒"
            } """
            response_data = {
                'input': message,
                'result': "检测结果: 仇恨检测模型目前并未实现"
            }

        else:
            response_data = {
                'input': message,
                'result': "无效的检测类型"
            }

    except Exception as e:
        response_data = {
            'input': message,
            'result': "后端服务器出错: " + str(e)
        }

    emit('response', response_data)



if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 8080), app)
    print('Server running at http://127.0.0.1:8080/')
    server.serve_forever()