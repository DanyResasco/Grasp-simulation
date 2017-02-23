import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
from klampt import vis 
from klampt.vis.glrobotprogram import *
from klampt.math import *
from klampt.model import collide
from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
import numpy as np
import math
import sys
from klampt.math import so3,se3,vectorops
import time


_has_numpy = False
_tried_numpy_import = False
np = None

def _try_numpy_import():
    global _has_numpy,_tried_numpy_import
    global np
    if _tried_numpy_import:
        return _has_numpy
    _tried_numpy_import = True
    try:
        import numpy as np
        _has_numpy = True
        #sys.modules['numpy'] = numpy
    except ImportError:
        print "klampt.model.sensing.py: Warning, numpy not available."
        _has_numpy = False
    return _has_numpy



def camera_to_images(camera,image_format='numpy',color_format='channels'):
    """Given a SimRobotSensor that is a CameraSensor, returns either the RGB image, the depth image, or both.

    If image_format='numpy' (default), returns numpy arrays.  Depending on the value of color_format, the RGB image
    either has shape (h,w,3) and dtype uint8 or (h,w) and dtype uint32.  Depth images as numpy arrays with
    shape (h,w).  Will fall back to 'native' if numpy is not available.

    If image_format='native', returns list-of-lists arrays in the same format as above

    If color_format='channels' (default), the RGB result is a 3D array with 3 channels corresponding to R, G, B
    values in the range [0,255].  If color_format='rgb' the result is a 2D array with a 32-bit integer channel
    with R,G,B channels packed in order XRGB.  If color_format='bgr' the result is similar but with order XBGR.

    (Note that image_format='native' takes up a lot of extra memory, especially with color_format='channels')
    """
    assert isinstance(camera,SimRobotSensor),"Must provide a SimRobotSensor instance"
    assert camera.type() == 'CameraSensor',"Must provide a camera sensor instance"
    w = int(camera.getSetting('xres'))
    h = int(camera.getSetting('yres'))
    has_rgb = int(camera.getSetting('rgb'))
    has_depth = int(camera.getSetting('depth'))
    measurements = camera.getMeasurements()
    if image_format == 'numpy':
        if not _try_numpy_import():
            image_format = 'native'
    rgb = None
    depth = None
    if has_rgb:
        if image_format == 'numpy':
            abgr = np.array(measurements[0:w*h]).reshape(h,w).astype(np.uint32)
            if color_format == 'bgr':
                rgb = abgr
            elif color_format == 'rgb':
                rgb = np.bitwise_or(np.bitwise_or(np.left_shift(np.bitwise_and(abgr,0x00000ff),16),
                                        np.bitwise_and(abgr,0x000ff00)),
                                        np.right_shift(np.bitwise_and(abgr,0x0ff0000), 16))
            else:
                rgb = np.zeros((h,w,3),dtype=np.uint8)
                rgb[:,:,0] =                np.bitwise_and(abgr,0x00000ff)
                rgb[:,:,1] = np.right_shift(np.bitwise_and(abgr,0x00ff00), 8)
                rgb[:,:,2] = np.right_shift(np.bitwise_and(abgr,0x0ff0000), 16)
        else:
            if color_format == 'bgr':
                rgb = []
                for i in xrange(h):
                    rgb.append([int(v) for v in measurements[i*w:(i+1)*w]])
            elif color_format == 'rgb':
                def bgr_to_rgb(pixel):
                    return ((pixel & 0x0000ff) << 16) | (pixel & 0x00ff00) | ((pixel & 0xff0000) >> 16)
                rgb = []
                for i in xrange(h):
                    rgb.append([bgr_to_rgb(int(v)) for v in measurements[i*w:(i+1)*w]])
            else:
                rgb = []
                for i in xrange(h):
                    start = i*w
                    row = []
                    for j in xrange(w):
                        pixel = int(measurements[start+j])
                        row.append([pixel&0xff,(pixel>>8)&0xff,(pixel>>16)&0xff])
                    rgb.append(row)
    if has_depth:
        start = (w*h if has_rgb else 0)
        if image_format == 'numpy':
            depth = np.array(measurements[start:start+w*h]).reshape(h,w)
        else:
            depth = []
            for i in xrange(h):
                depth.append(measurements[start+i*w:start+(i+1)*w])
    if has_rgb and has_depth:
        return rgb
        # ,depth
    elif has_rgb:
        return rgb
    elif has_depth:
        return depth
    return None