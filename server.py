from flask import Flask, render_template, jsonify, request, send_file

import io 
import base64 
from PIL import Image

import threading
import time
from queue import Queue,Empty
from maskDetection import inference
from load_model.pytorch_loader import load_pytorch_model

app = Flask(__name__, template_folder="/templates/")

requests_queue = Queue()
BATCH_SIZE = 0
CHECK_INTERVAL = 0.1

b = time.time()
model = load_pytorch_model('models/model360.pth') 
print("model load time : ", time.time()-b)
# def handle_requests_by_batch():
#     while True:
#         requests_batch = []
#         while not (len(requests_batch) >= BATCH_SIZE):
#             try:
#                 requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
#             except Empty:
#                 continue

#             for request in requests_batch:
#                 request['output'] = run(request['input'][0])

# threading.Thread(target=handle_requests_by_batch).start()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "ok", 200


@app.route("/detect", methods=["POST"])
def detect():
    image = Image.open(request.files['image'].stream).convert('RGB')

    # req = {
    #     'input': [image]
    # }

    # requests_queue.put(req)

    # while 'output' not in req:
    #     time.sleep(CHECK_INTERVAL)
    
    # result = req['output']
    result = inference(model, image, target_shape=(360, 360))

    if 'msg' in result:
        return jsonify(result['msg']), 500

    # image = result[1]
    # im_pil = Image.fromarray(image)
    # iio = io.BytesIO()
    # im_pil.save(iio, 'PNG')
    # iio.seek(0)

    image = get_response_image(result[1])
    print(result[0])
    return jsonify({'info':str(result[0]),'img':image}), 200
    # return send_file(iio, mimetype='image/png')
    
def run(image):
    
    try:
        result = inference(model, image, target_shape=(180, 180))
    except:
        return {'msg': 'Dectection Error'}

    return result

def get_response_image(image):
    im_pil = Image.fromarray(image)
    img_io = io.BytesIO()
    im_pil.save(img_io, 'PNG')

    encoded_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
    encoded_img = 'data:image/png;base64,' + encoded_img
    return encoded_img

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
