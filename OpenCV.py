import cv2, numpy as np

global video
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class ChrisCam(object):
    def __init__(self,camera=0):
        self.cap = cv2.VideoCapture(camera)
        self.image_type = 0
    
    def setSize(self,width = 640, height = 480):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        return self.cap.isOpened()

    def snap(self):
        ret,frame = self.cap.read()
        return ret, frame
    
    def toLV(self,frame):
        height,width = frame.shape[0:2]
        self.rectangle = (0,0,width,height)
        image_depth = 24
        colors = np.array([])
        framelist = frame.ravel().tolist()
        reply = (self.image_type, image_depth,framelist,np.array([]).tolist(),colors.tolist(),self.rectangle)
        return reply

    def toCV(self,LVimage):      # takes around 30 ms per call (150ms for Brian's implementation?)
        rows,cols = LVimage[5][3],LVimage[5][2]
        array = np.asarray(LVimage[2],dtype='uint8')
        img = array.reshape(rows,cols,3)
        return img

    def flip(self,LVframe,direction):
        image = video.toCV(LVframe) 
        image = cv2.flip(image,direction)
        return video.toLV(image)

    def face(self,LVframe):
        frame = video.toCV(LVframe)
        mid = frame.shape[1]/2
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        try:
            faces = face_cascade.detectMultiScale(gray, 1.3, 3)
        except:
            faces = [(10,10,10,10),(10,460,10,10),(620,10,10,10),(620,460,10,10)] # indicates loading failure
        centers,weights = [],[]
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            centers.append(x+w/2)
            weights.append(w*h+1)
        try:
            cx = np.average(centers,weights=weights)
        except:
            cx = mid
        return (video.toLV(frame),mid-cx)
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        return True

def initCam(camera = 0, width = 640, height = 480):
    global video
    video = ChrisCam(camera)
    return video.setSize(width,height)

def snap():
    ret, frame = video.snap()
    if not ret: return video.toLV(np.full((480,640,3),150))
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return video.toLV(frame)

def flipH(frame):
    return video.flip(frame,1)

def flipV(frame):
    return video.flip(frame,0)

def faceDetect(frame):
    return video.face(frame)[0]

def faceTrack(frame):
    return video.face(frame)

def CannyEdge(LVframe, sigma=0.33):
    gray = cv2.cvtColor(video.toCV(LVframe), cv2.COLOR_BGR2GRAY)
    mid = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * mid))
    upper = int(min(255, (1.0 + sigma) * mid))
    edges = cv2.Canny(gray, lower, upper)
    img = np.dstack((edges,edges,edges))
    return video.toLV(img)

def closeCam():
    cv2.destroyAllWindows()
    video.close()

def debug(frame):
    return "alter this return statement for debugging purposes"