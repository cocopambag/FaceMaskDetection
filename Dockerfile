FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
RUN pip install --upgrade pip 
RUN apt-get install ffmpeg -y

COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt 

COPY . .

CMD "bash"
