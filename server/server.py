import os

from flask import Flask, jsonify, abort, make_response, request
from flask_httpauth import HTTPBasicAuth
from core import matching, restudy, error_counter
from prometheus_client import Gauge, start_http_server, Histogram, Counter

app = Flask(__name__)

start_http_server(8000)

auth = HTTPBasicAuth()

users = {
    os.getenv("SERVER_USER"): os.getenv("PASSWORD"),
}

class Collector:
    def __init__(self, ):
        self.metric_request_counter = Counter('requests', 'count of outer requests for prediction')
        self.metric_learning_counter = Counter('learning', 'count of outer requests for learning')
        self.metric_error_counter = Counter('error', 'count of errors in lerning')
        self.metric_request_time = Histogram('request_processing_seconds',
                       'Time spent processing request')
        self.metric_learning_time = Histogram('learning_processing_seconds',
                       'Time spent learning request')

    def collect_info(self, accuracy, f1, precision, recall, sample_count):
        self.metric_request_counter.set(request_counter)
        self.metric_learning_counter.set(learning_counter)
        self.metric_error_counter.set(error_counter)


collector = Collector()
request_counter = 0
learning_counter = 0


@collector.metric_request_time.time()
@app.route('/user/<int:userid>', methods=['GET'])
@auth.login_required
def get_task(userid):
    if type(userid) is int:
        request_counter += 1
        items = list(matching(userid).itemid)
    else:
        abort(404)
    return jsonify({'itemid_1': items[0],
                    'itemid_2': items[1],
                    'itemid_3': items[2]})


@collector.metric_learning_time.time()
@app.route('/restudy', methods=['POST'])
@auth.login_required
def create_task():
    if request.json['command'] == 'restudy' and request.json['password'] == 123:
        learning_counter += 1
        task = restudy()
    else:
        abort(400)
    return jsonify({'result': task}), 200


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)

if __name__ == '__main__':
    app.run(debug=True)
