from proton_ai import ProtonAI
import argparse
import cv2

CAMERA_ID = 0

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--operation", required=True, help="Select the operation to be performed.")
ap.add_argument("-i", "--image", nargs='*', help="Input preprocessed image.")
args = vars(ap.parse_args())

def capture_image(device_id, show=False):
    cam = cv2.VideoCapture(device_id)
    ret, img = cam.read()
    return img

if args['operation'] == 'calibrate':
    if args['image'] is not None:
        img = args['image']
    else:
        img = capture_image(CAMERA_ID)
    
else:
    if args['image'] is not None:
        img = args['image']
    else:
        img = capture_image(CAMERA_ID)

# Initialize the AI.
ai = ProtonAI(img, args['operation'])