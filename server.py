from flask import Flask, render_template, jsonify, request, send_file

import io 
import base64 
import uuid
import os
from PIL import Image

import threading
import time
from queue import Queue,Empty
from maskDetection import inference, run_on_video
from load_model.pytorch_loader import load_pytorch_model
from utils.meta import toH264

app = Flask(__name__, template_folder="./templates/")

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

b = time.time()
model = load_pytorch_model('models/model360.pth') 
print("model load time : ", time.time()-b)

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for request in requests_batch:
                request['output'] = run(request['input'][0], request['input'][1])

threading.Thread(target=handle_requests_by_batch).start()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "ok", 200


@app.route("/detect-image", methods=["POST"])
def detect_image():
    if requests_queue.qsize() > BATCH_SIZE: 
        return jsonify({'msg': 'Too Many Requests'}), 429

    # read Image RGB
    image = Image.open(request.files['image'].stream).convert('RGB')

    # for Queue
    req = {
        'input': [image,"image"]
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    result = req['output']

    if 'msg' in result:
        return jsonify(result['msg']), 500

    # convert image stream to base64 URL
    image = get_base64URL(result[1])
    return jsonify({'info':str(result[0]),'img':image}), 200

@app.route("/detect-video", methods=["POST"])
def detect_video():
    if requests_queue.qsize() > BATCH_SIZE: 
        return jsonify({'msg': 'Too Many Requests'}), 429

    try:
        video = request.files['video']
        video_id = str(uuid.uuid4())

        video.save(video_id)
    except Exception as e:
        print(e)
        return jsonify({'msg': 'Invalid file'}), 400
    
    req = {
        'input':[video_id, "video"]
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    result = req['output']

    final = open(result,'rb')

    video_remove(result)

    if 'msg' in result:
        return jsonify(result['msg']), 500

    return send_file(final, mimetype="video/mp4")
    
def run(input_file, mode):

    if mode == "image":
        return run_image(input_file)
    else:
        return run_video(input_file)

def run_image(image):
    try:
        result = inference(model, image, target_shape=(360, 360))
    except:
        return {'msg': 'Dectection Error'}

    return result

def run_video(video_id):
    
    video_output = video_id + '_result.mp4'
    try:
        result = run_on_video(model, video_id, video_output, conf_thresh=0.5)

        if result['msg'] != 'Success':
            video_remove(video_id)
            return jsonify({'msg' : result['msg']}), 500

        moditied_video = toH264(video_output)
        video_remove(str(video_id))
        video_remove(video_output)

    except Exception as e:
        print(e)
        return jsonify({'msg': 'Dectection Error'})
    
    return moditied_video


def get_base64URL(image):
    im_pil = Image.fromarray(image)
    img_io = io.BytesIO()
    im_pil.save(img_io, 'PNG')

    encoded_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
    encoded_img = 'data:image/png;base64,' + encoded_img
    return encoded_img

def video_remove(video_id):
    if os.path.exists(video_id):
        os.remove(video_id)
        return True
    return False

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80) 
