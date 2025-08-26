from flask import Flask, jsonify, abort, make_response, request
from core import matching, restudy

app = Flask(__name__)


@app.route('/user/<int:userid>', methods=['GET'])
def get_task(userid):
    if type(userid) is int:
        items = list(matching(userid).itemid)
    else:
        abort(404)
    return jsonify({'itemid_1': items[0],
                    'itemid_2': items[1],
                    'itemid_3': items[2]})


@app.route('/restudy', methods=['POST'])
def create_task():
    if request.json['command'] == 'restudy' and request.json['password'] == 123:
        task = restudy()
    else:
        abort(400)
    return jsonify({'result': task}), 200


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=False)
