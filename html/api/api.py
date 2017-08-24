import numpy as np
import random
import tensorflow as tf
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import helper

app = Flask(__name__)
CORS(app)

def get_tensors(loaded_graph):
    return loaded_graph.get_tensor_by_name("input:0"), loaded_graph.get_tensor_by_name("initial_state:0"), loaded_graph.get_tensor_by_name("final_state:0"), loaded_graph.get_tensor_by_name("probs:0")


def pick_word(probabilities, int_to_vocab):
    possibilities = []
    for ix, prob in enumerate(probabilities):
        if prob >= .1:
            possibilities.append(int_to_vocab[ix])
    rand = np.random.randint(0, len(possibilities))

    return str(possibilities[rand])


def massage_results(gen_sentences, token_dict):
    
    wtext = ' '.join(gen_sentences)

    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        wtext = wtext.replace(' ' + token.lower(), key)

    wtext = wtext.replace(' [', '[')
    wtext = wtext.replace('[ ', '[')
    wtext = wtext.replace(' ]', ']')
    wtext = wtext.replace('] ', ']')
    wtext = wtext.replace('[', '')
    wtext = wtext.replace(']', '')
    wtext = wtext.replace('(', ' (')
    wtext = wtext.replace('( ', ' (')
    wtext = wtext.replace(' )', ')')
    wtext = wtext.replace('  ', ' ')    
    wtext = wtext.replace('  ', ' ')    
    wtext = wtext.replace('\n', '|')

    wtext = wtext.split("|")   

    return wtext


def parse_postget(f):
    @wraps(f)
    def wrapper(*args, **kw):
        try:
            d = dict((key, request.values.getlist(key) if len(request.values.getlist(
                key)) > 1 else request.values.getlist(key)[0]) for key in request.values.keys())
        except BadRequest as e:
            raise Exception("Payload must be a valid json. {}".format(e))
        return f(d)
    return wrapper

@app.route('/')
def index():
	return "wodbot rocks!"
	
@app.route('/model', methods=['GET', 'POST'])
@parse_postget


def apply_model(d):
    gen_length = 30
    if request.method == 'POST':
        content = request.json
        gen_length = content['gen_length']

    _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

    prime_word = random.choice(int_to_vocab)
    print (prime_word)
        
    seq_length, load_dir = helper.load_params()

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

        # Sentences generation setup
        gen_sentences = [prime_word]
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
        print (gen_sentences)   
        
        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

            gen_sentences.append(pred_word) 

        data = massage_results(gen_sentences, token_dict)
		if (len(data) > 2):
        	wod_return = random.choice(data[1:-1])  
		else:
			wod_return = data[0]
    
    return jsonify(output=wod_return)

if __name__ == '__main__':
    app.run(debug=True)
