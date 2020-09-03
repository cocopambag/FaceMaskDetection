import subprocess, re
import os

def checkRotate(video_path):
    cmd = 'ffmpeg -i ' + video_path

    p = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        close_fds=True,
        shell=True
    )
    stdout, stderr = p.communicate()
    stderr = stderr.decode('utf-8')
    reo_rotation = re.compile('rotate\s+:\s(?P<rotation>.*)')
    match_rotation = reo_rotation.search(stderr)
    rotation = match_rotation.groups()[0]
    return int(rotation)

def toH264(video_path):
    modified_name = video_path + "_modified.mp4"
    cmd = 'ffmpeg -i ./' + video_path + ' -c:v libx264 ' +  modified_name

    
    os.system(cmd)

    return modified_name