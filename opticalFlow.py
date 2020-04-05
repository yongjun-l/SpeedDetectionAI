import numpy as np
import cv2

def opticalFlowDenseA(prvs, next):
    prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(next)
    hsv[...,1] = 255
    ret, frame2 = cap.read()
    next = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)

    #obtain dense optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 2, 5, 1.2, 0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    #hue corresponds to direction
    hsv[...,0] = ang*180/np.pi/2

    #value corresponds to magnitude
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    #convert hsv to bgr
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return bgr

def opticalFlowDenseB(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    
    hsv = np.zeros((66, 220, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
 
    # Flow Parameters
    # flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow



cap = cv2.VideoCapture(cv2.samples.findFile("test.mp4"))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('optical_flow_test.mp4', fourcc, 15, (640,160))

while 1:
    ret, frame1 = cap.read()
    ret1, frame2 = cap.read()

    if ret1:
        frame1 = frame1[190:360, 0:640]
        frame2 = frame2[190:360, 0:640]

        dim = (220, 66)
        frame1 = cv2.resize(frame1, dim, interpolation = cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, dim, interpolation = cv2.INTER_AREA)


        rgb_flow = opticalFlowDenseA(frame1, frame2)
        dim = (640, 160)
        rgb_flow = cv2.resize(rgb_flow, dim, interpolation = cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow("frame2", rgb_flow)
        cv2.imshow("frame1", frame2)
        k = cv2.waitKey(30) & 0xff

        out.write(rgb_flow)
    else:
        break

out.release()