from flask import Flask, request, send_file
from io import BytesIO
import cv2
import numpy
import os
import Cards

app = Flask(__name__)

net_location = os.environ.get('NET_DIR')
net = cv2.dnn.readNet("{}/yolocards_608.weights".format(net_location), "{}/yolocards.cfg".format(net_location))

classes = []
with open("{}/cards.names".format(net_location), "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = numpy.random.uniform(0, 255, size=(len(classes), 3))

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = cards.load_ranks( path + '/Card_Imgs/')
train_suits = cards.load_suits( path + '/Card_Imgs/')

@app.route('/api/ocr', methods=['POST'])
def process():
    filestr = request.files['image'].read()

    byte_file = BytesIO(filestr)
    data = numpy.frombuffer(byte_file.read(), numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return 'hello'

if __name__ == "__main__":
    app.run(debug=True)
