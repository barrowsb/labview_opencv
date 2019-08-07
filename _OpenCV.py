# Tufts University Center for Engineering Education and Outreach
# Summer 2019
# Main Contributors: Brian Barrows, Chris Rogers
# This is trial Python code used in developing the OpenCV SubVIs.

import cv2, numpy as np
from timeit import default_timer as timer

global video

face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades\haarcascade_smile.xml')
profile_cascade = cv2.CascadeClassifier('cascades\haarcascade_profileface.xml')
nose_cascade = cv2.CascadeClassifier('cascades\haarcascade_nose.xml')

class ChrisCam(object):
    def __init__(self,camera = 'c:/fred.avi'):
        self.cap = cv2.VideoCapture(camera)
        self.image_type = 0
    
    def size(self,width = 320, height = 240):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        return self.cap.isOpened()
    
    def snap(self, env, show=False, fx=1, fy=1):
        ret,frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=fx, fy=fy)
            if show: cv2.imshow('image',frame)
        return ret, frame
    
    def colorSwap(self,frame,plane,env):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if plane=='Gray':
            frame = gray
        elif env=='LV':
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if plane=='R':
                frame[:,:,1:]=0
            elif plane=='G':
                frame[:,:,0],frame[:,:,2]=0,0
            elif plane=='B':
                frame[:,:,:2]=0
        elif env=='python':
            if plane=='R':
                frame[:,:,:2]=0
            elif plane=='G':
                frame[:,:,0],frame[:,:,2]=0,0
            elif plane=='B':
                frame[:,:,1:]=0
        return gray,frame
    
    def harris(self, frame, gray, thresh):
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        try:
            frame[dst>thresh*dst.max()]=[0,0,255]
        except ValueError:
            frame[dst>thresh*dst.max()]=255
        return frame
    
    def face(self, frame, gray, levels):
        faces = face_cascade.detectMultiScale(gray, 1.3, levels)
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 6)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 6)
            for (nx,ny,nw,nh) in noses:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),1)
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 6)
            for (sx,sy,sw,sh) in smiles:
                cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),1)
        return frame
    
    def rotate(self, img, rotation):
        rows,cols = img.shape[0:2]
        rotationMatrix = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1)
        frame = cv2.warpAffine(img,rotationMatrix,(cols,rows))
        return frame
    
    def toLV(self,frame,plane = 'RGB'):
        height,width = frame.shape[0:2]
        self.rectangle = (0,0,width,height)  # this is the LabVIEW rectangle format
        if plane=='Gray':
            image_depth = 8
            colors = np.array([ (i | i<<8 | i<<16) for i in range(256)])
            if width%2: frame = np.hstack((frame,np.zeros((height,1))))
        else:
            image_depth = 24
            colors = np.array([])
        framelist = frame.ravel().tolist()
        reply = (self.image_type, image_depth,framelist,np.array([]).tolist(),colors.tolist(),self.rectangle)
        return reply
 
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        return True

def initCam(camera = 'c:/fred.avi',width = 640, height = 480):
    global video
    video = ChrisCam(camera)
    return video.size(width,height)

def snap(env='python', plane='RGB', show=False, rotation=90, fx=1, fy=1, harris=False, thresh=0.05, face=False, levelweights=5):
    ret, frame = video.snap(env,show,fx,fy)
    if not ret: return
    gray,frame = video.colorSwap(frame,plane,env)
    if harris: frame = video.harris(frame,gray,thresh)
    if face: frame = video.face(frame,gray,levelweights)
    frame = video.rotate(frame,rotation)
    if env=='LV': frame = video.toLV(frame,plane)
    return frame

def closeCam():
    cv2.destroyAllWindows()
    return video.close()

def pytest(cam=0, plane='RGB', rotation=0, fx=1, fy=1, harris=False, thresh=0.05, face=False, levelweights=4):
    initCam(cam)
    now = 0
    fps = [0]*10
    while True:
        past = now
        now = timer()
        dt = round(1000*(now-past))
        fps.append(1000/dt)
        fps.pop(0)
        print('dt:  ',dt,'	fps:  ',round(np.mean(fps)))
        feed = snap(plane=plane,rotation=rotation,fx=fx,fy=fy,harris=harris,thresh=thresh,face=face,levelweights=levelweights)
        if feed is None: continue
        cv2.imshow('Press any key to exit...',feed)
        if cv2.waitKey(1)>=0: break
    print('Pytest ended')
    closeCam()
