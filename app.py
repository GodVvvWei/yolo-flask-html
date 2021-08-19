

#作者：小约翰啊伟
#B站主页：https://space.bilibili.com/420694489
#源码下载：https://github.com/GodVvvWei/yolo-flask-html


from flask import Flask,render_template,Response
import cv2

from models.experimental import attempt_load
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device

app = Flask(__name__)
user, pwd, ip = "admin", "123456zaQ", "[192.168.100.196]"
from camera_ready import detect


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流（海康摄像头）
        self.count = 0
        self.video = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, 1))
        #大华摄像头
        #self.video = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))

        self.weights, imgsz = 'yolov5s.pt', 640
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
    def __del__(self):
        self.video.release()

    def get_frame(self):

        for i in range(50):
            success, image = self.video.read()
        image= detect(source=image,half=self.half,model=self.model,device=self.device,imgsz=self.imgsz,stride=self.stride)

        ret,jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


@app.route('/xyhaw')
def xyhaw():
    return render_template('xyhaw.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
