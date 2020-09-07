from flask import Flask, render_template, jsonify, request, send_file

import io 
import base64 
import uuid
import os
from PIL import Image
import numpy as np 
import shutil

import threading
import time
from queue import Queue,Empty
from maskDetection import inference, run_on_video
from load_model.pytorch_loader import load_pytorch_model
from utils.meta import toH264

DATA_PATH = './data/'

app = Flask(__name__, template_folder="./templates/")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024 # 15MB

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

startTime = time.time()
model = load_pytorch_model('models/model360.pth') 
print("model load time : ", time.time()-startTime)

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
    print(requests_queue.qsize())
    if requests_queue.qsize() > BATCH_SIZE: 
        return jsonify({'msg': 'Too Many Requests'}), 429

    try:
        # read Image
        image = Image.open(request.files['image'].stream).convert('RGB')
    except Exception as e:
        print(e)
        return jsonify({'msg':'Invalid file'}), 400

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

    video_id = str(uuid.uuid4())
    file_path = os.path.join(DATA_PATH, video_id)
    os.makedirs(file_path, exist_ok=True)

    try:
        video = request.files['video']
        video_path = os.path.join(file_path, "original.mp4")
        video.save(video_path)
    except Exception as e:
        print(e)
        return jsonify({'msg': 'Invalid file'}), 400
    
    # for Queue
    req = {
        'input':[file_path, "video"]
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    result = req['output']

    if 'msg' in result:
        return jsonify(result['msg']), 500

    final = open(result,'rb')

    shutil.rmtree(file_path)

    return send_file(final, mimetype="video/mp4")
    
def run(input_file, mode):

    if mode == "image":
        return run_image(input_file)
    else:
        return run_video(input_file)

def run_image(image):
    try:
        result = inference(model, image, target_shape=(360, 360),mode=0)
    except:
        return {'msg': 'Dectection Error'}

    return result

def run_video(file_path):

    original_video_path = os.path.join(file_path, "original.mp4")
    resize_video_path = os.path.join(file_path, "resize.mp4")
    result_video_path = os.path.join(file_path, "result.mp4")

    try:
        # convert video to 15fps 
        width_resize = 480
        os.system(
            "ffmpeg -hide_banner -loglevel warning -ss 0 -i '{}' -t 10 -filter:v scale={}:-2 -r 15 -c:a copy '{}'".format(
                os.path.abspath(original_video_path), width_resize, os.path.abspath(resize_video_path)))
    except Exception as e:
        print(e)
        return jsonify({'msg': 'Resizing fail'}), 500
    
    try:
        result = run_on_video(model, resize_video_path, result_video_path, conf_thresh=0.5)

        if result['msg'] != 'Success':
            shutil.rmtree(file_path)
            return jsonify({'msg' : result['msg']}), 500

        moditied_video = toH264(result_video_path)

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


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
    # app.run(host='0.0.0.0', port=80) 
