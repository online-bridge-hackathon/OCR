from flask import Flask, request, send_file
from io import BytesIO
import cv2
import numpy

app = Flask(__name__)

@app.route('/api/ocr', methods=['POST'])
def process():
    filestr = request.files['image'].read()

    byte_file = BytesIO(filestr)
    data = numpy.frombuffer(byte_file.read(), numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return 'hello'

if __name__ == "__main__":
    app.run(debug=True)
