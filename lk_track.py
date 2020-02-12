#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from copy import copy 

import video
from common import anorm2, draw_str
from time import sleep
from shutil import rmtree as remove
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 1,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.5,
                       minDistance = 15,
                       blockSize = 1 )

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.1,
                       minDistance = 15)
class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 15
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        self.check = False
        self.color = [(255,0,0), (0,255,0),(0,0,255),(0,0,0),(255,255,255),(255,255,0),(0,255,255),(255,0,255)] *100
        self.color_id =0
        self.gps = [0] 
        self.distCoeffs = np.array([ -0.38846448, 0.12976164, 0.0, 0.0, 0.0 ])
        self.cameraMatrix = np.array([[ 903.69745, 0.0, 634.7149],
                                [0.0, 914.1482, 384.99857 ],
                                [0.0,     0.0,    1.0     ]                          
                                                                    ])






    

    def undistort(self, image, cameraMatrix,distCoeffs, scale = 0) :
        """
        Input :
        image : the image to be processed
        cameraMatrix :  3x3 ndarray 
        distCoeffs   : 1x5 ndarray conatining the distortion coefficients, they can be 4, 5, 8,12, or 14. (in our case )
        undistortedImage : the image after processing, without the lens distortions
        """
            
        
        #getting the images size
        h, w = image.shape[:2]
        # from the intrinsic get the optimum 
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))
        # perform the undistortion tranformation on the image
        undistortedImage = cv.undistort(image,cameraMatrix, distCoeffs, scale, newCameraMatrix )
        x,y,w,h = roi #
        undistortedImage  = undistortedImage[y:y+h, x:x+w]
        undistortedImage = cv.resize(undistortedImage,(1280,720))

        return undistortedImage 



    def run(self):
        jump = 5999
        _frame_id = jump
        self.track_id = []
        
        while True:

            
            if _frame_id == jump :
                self.cam.set(cv.CAP_PROP_POS_FRAMES, _frame_id) 
            elif _frame_id == 6484 :
                _frame_id = jump - 1
            _ret, frame = self.cam.read()
            frame = self.undistort(frame,self.cameraMatrix, self.distCoeffs)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray

                # p0 : last points seen and extracted in preview(last) image 
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2) 

                # p1 : points matching from previous image and the new one, using last points p0
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) 

                # p0r : points matching from new image and the previous image, using new points p1 
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params) 

                # check the axis distance between those points (Not actulally a distance )
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                
                # requirement to keep following a track
                good = d < 0.01
                new_tracks = []
                
               
                # TODO how to get point already seen ?

                for tr, (x, y), good_flag,gps in copy(zip(self.tracks, p1.reshape(-1, 2), good, self.gps)):
                    if not good_flag:
                        continue

                    tr.append((x, y))
                    
                    if len(tr) > self.track_len:
                        del tr[0]
            
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
               

                self.gps = [0] * p.shape[0]
                
                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, self.color[self.color_id]) #(0, 255, 0)
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:

                # mask is the region of interest where the feature should be detected
                mask = np.zeros_like(frame_gray)

                mask[:] = 255

                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)

                # Make sure thats the new feature are really good features
                # extract good features to track 
                p = cv.goodFeaturesToTrack(frame_gray, mask=mask ,maxCorners = 500,qualityLevel = 0.1, minDistance = 15)
        
                # remove those who don't respect the orb feature description requirement
                orb =  cv.ORB_create()
                kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in p]
                kps, _ = orb.compute(frame_gray, kps)
                p = np.array([[(kp.pt[0], kp.pt[1])] for kp in kps], dtype=np.float32) 

                
                
                
                self.gps = [1] * p.shape[0]
               
                
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

                
                

                self.color_id += 1
            self.frame_idx += 1
            _frame_id += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break
            if ch & 0xFF == ord('q') :
                break
            
def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    App(video_src).run()
    remove("./__pycache__")
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
