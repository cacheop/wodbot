import numpy as np
import tensorflow as tf
from functools import wraps
from flask import Flask, request, jsonify
import helper

app = Flask(__name__)

gen_length = 380
prime_word = 'shots'

def get_tensors(loaded_graph):
    return loaded_graph.get_tensor_by_name("input:0"), loaded_graph.get_tensor_by_name("initial_state:0"), loaded_graph.get_tensor_by_name("final_state:0"), loaded_graph.get_tensor_by_name("probs:0")


def pick_word(probabilities, int_to_vocab):
    possibilities = []
    for ix, prob in enumerate(probabilities):
        if prob >= .05:
            possibilities.append(int_to_vocab[ix])
    rand = np.random.randint(0, len(possibilities))

	return str(possibilities[rand])
	

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

@app.route('/model', methods=['GET', 'POST'])
@parse_postget
def apply_model(d):
	_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

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

	    # Remove tokens
	    tv_script = ' '.join(gen_sentences)
	    for key, token in token_dict.items():
	        ending = ' ' if key in ['\n', '(', '"'] else ''
	        tv_script = tv_script.replace(' ' + token.lower(), key)
	    tv_script = tv_script.replace('\n ', '\n-----------------\n')
	    tv_script = tv_script.replace('( ', '(')
	
    return jsonify(output=tv_script)

if __name__ == '__main__':
    app.run(debug=True)
