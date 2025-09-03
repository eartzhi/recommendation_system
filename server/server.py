import os

from flask import Flask, jsonify, abort, make_response, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from core import matching, relearning, error_counter, users, top_products
from prometheus_client import make_wsgi_app, Histogram, Gauge

app = Flask(__name__)

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': make_wsgi_app()})

auth = HTTPBasicAuth()

# print(os.getenv("SERVER_USER"), os.getenv("PASSWORD"))

users = {
    os.getenv("SERVER_USER"): os.getenv("PASSWORD"),
}




class Collector:
    def __init__(self, ):
        self.metric_request_counter = Gauge('requests', 'count of outer requests for prediction')
        self.metric_learning_counter = Gauge('learning', 'count of outer requests for learning')
        self.metric_error_counter = Gauge('error', 'count of errors in lerning')
        self.metric_request_time = Histogram('request_processing_seconds',
                       'Time spent processing request')
        self.metric_learning_time = Histogram('learning_processing_seconds',
                       'Time spent learning request')

    def collect_info(self, request_counter, learning_counter, error_counter):
        self.metric_request_counter.set(request_counter)
        self.metric_learning_counter.set(learning_counter)
        self.metric_error_counter.set(error_counter)


collector = Collector()
request_counter = 0
learning_counter = 0
# items = [1,2,3]

collector.collect_info(request_counter, learning_counter, error_counter)

@collector.metric_request_time.time()
@app.route('/user/<int:userid>', methods=['GET'])
@auth.login_required
def get_task(userid):
    global request_counter
    if type(userid) is int:
        request_counter += 1
        if userid in users:
            items = top_products
        else:
            items = list(matching(userid).itemid)
    else:
        abort(404)
    return jsonify({'itemid_1': items[0],
                    'itemid_2': items[1],
                    'itemid_3': items[2]})


@collector.metric_learning_time.time()
@app.route('/relearning', methods=['POST'])
@auth.login_required
def create_task():
    global learning_counter
    if request.json['command'] == 'restudy' and request.json['password'] == os.getenv("RES_PASSWORD"):
        learning_counter += 1
        task = relearning(request.json['parameters'])
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
    app.run(host='0.0.0.0', port=5000, debug=False)
