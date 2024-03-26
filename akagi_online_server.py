import json
from flask import Flask, request, jsonify, make_response

from model import get_engine

import numpy as np
import gzip

app = Flask(__name__)
engine = get_engine()


@app.route('/react_batch', methods=['POST'])
def react_batch():
    # Decompress the data using gzip
    data = request.data
    data = json.loads(gzip.decompress(data))

    if not data or 'obs' not in data or 'masks' not in data:
        return jsonify({'error': 'Bad request'}), 400

    json_obs = data['obs']
    json_masks = data['masks']

    obs = [np.array(o, dtype=np.float32) for o in json_obs]
    masks = [np.array(m) for m in json_masks]


    global engine
    actions, q_out, masks, is_greedy = engine.react_batch(obs, masks, None)
    

    response = {
        'actions': actions,
        'q_out': q_out, 
        'masks': masks,
        'is_greedy': is_greedy
    }
    
    ans = (jsonify(response), 200)

    return ans

@app.route('/check', methods=['POST'])
def check():
    return jsonify({'result': 'success'}), 200
    

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5003)
