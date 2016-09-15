from flask import Flask, jsonify, request, render_template

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import tensorflow as tf
import os

def get_image_score(image_path):
    full_path = os.getcwd()+'/app/static/images/'+image_path
    image_data = tf.gfile.FastGFile(full_path, 'rb').read()

    label_lines = sorted([line.rstrip() for line in tf.gfile.GFile('/Users/Saniya/tf_files/retrained_labels1.txt')])

    with tf.gfile.FastGFile('/Users/Saniya/tf_files/retrained_graph1.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    results = []

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        with tf.Graph().as_default():
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in range(0, 13):#top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            results.append({
                "axis": human_string.title(),
                "value":score
                })
            print('%s (score = %.5f)' % (human_string, score))
    return results


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)

# Get an example and return it's score from the predictor model
@app.route("/", methods=["GET", "POST"])
def score():
    # """
    # When A POST request with json data is made to this uri,
    # Read the example from the json, predict probability and
    # send it with a response
    # """

    image_path = request.form.get("image_path", "")
    if image_path:
        results = [get_image_score(image_path)]
    else:
        results = [[]]
    return render_template("index.html", image_path=image_path, results=results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
