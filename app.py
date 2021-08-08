

#作者：小约翰啊伟
#B站主页：https://space.bilibili.com/420694489
#源码下载：https://github.com/GodVvvWei/yolo-flask-html
from flask import Flask,render_template,Response
import cv2
app = Flask(__name__)
user, pwd, ip = "admin", "123456zaQ", "[192.168.100.196]"
from camera_ready import detect



# mp.set_start_method(method='spawn')
# queue = mp.Queue(maxsize=2)
#
# def run_single_camera():
#     # user_name, user_pwd, camera_ip = "admin", "admin123456", "172.20.114.196"
#     user_name, user_pwd, camera_ip = "admin", "123456zaQ", "[192.168.100.196]"
#
#       # init
#
#
#     p1 = mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip))
#     p2 = mp.Process(target=image_get, args=(queue,))
#     p3= mp.Process(target=run,args=(queue,))
#     p1.start()
#     p1.join()
#     p2.start()
#     p2.join()
#     p3.start()
#     p3.join()
#
# def image_put(q, user, pwd, ip, channel=1):
#     cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))
#     if cap.isOpened():
#         print('HIKVISION')
#     else:
#         cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
#         print('DaHua')
#     while True:
#         q.put(cap.read()[1])
#         q.get() if q.qsize() > 1 else time.sleep(0.01)
# def image_get(q):
#     while True:
#         image = q.get()
#         image = detect(source=image)
#
#         ret, jpeg = cv2.imencode('.jpg', image)
#         global a
#         a = jpeg.tobytes()

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流（海康摄像头）
        self.count = 0
        self.video = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, 1))
        #大华摄像头
        #self.video = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))


    def __del__(self):
        self.video.release()

    def get_frame(self):
        # self.count += 1
        success, image = self.video.read()
        image = detect(source=image)
        # if self.count >10:
        #     image = detect(source=image)
        #     self.count = 0

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


@app.route('/xyhaw')
def xyhaw():
    return render_template('xyhaw.html')


# def gen(camera):
def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        # image = q.get()
        # image = detect(source=image)
        # ret, jpeg = cv2.imencode('.jpg', image)
        # frame = jpeg.tobytes()
        # frame = a
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    # return Response(gen(VideoCamera()),
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# def run(q):
#     app.run(debug=True)
if __name__ == '__main__':
    # app.run(host='0.0.0.0',debug=True)
    app.run(debug=True)
    # run_single_camera()

