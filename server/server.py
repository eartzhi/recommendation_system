from flask import Flask, jsonify, abort, make_response, request
from flask_httpauth import HTTPBasicAuth
from core import matching, restudy

app = Flask(__name__)

auth = HTTPBasicAuth()

users = {
    "shop": "password",

}

@app.route('/user/<int:userid>', methods=['GET'])
@auth.login_required
def get_task(userid):
    if type(userid) is int:
        items = list(matching(userid).itemid)
    else:
        abort(404)
    return jsonify({'itemid_1': items[0],
                    'itemid_2': items[1],
                    'itemid_3': items[2]})


@app.route('/restudy', methods=['POST'])
@auth.login_required
def create_task():
    if request.json['command'] == 'restudy' and request.json['password'] == 123:
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
    app.run(debug=False)
