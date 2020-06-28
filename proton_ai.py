from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from utils import *
import numpy as np
import argparse
import imutils
import json
import cv2
import os

POSITION_TOLERANCE = 0.1
WHOLE_AREA_TOLERANCE = 0.3
OUTER_CIRCLE_AREA_TOLERANCE = 0.2

class ProtonAI:
    def __init__(self, image, operation='test'):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1920, 1080)

        self.REFERENCE_WIDTH = 173                      # Reference Width
        self.pixelsPerMetric = None                     # Pixels per metric Ratio.

        if operation == 'calibrate':
            self.calibrate_system(image)
            return

        if os.path.isfile('data/calibrate_data.json') == False:
            print("\n[ERROR] Calibration data not found. Please calibrate the system first.\n")
            return False

        # Load the calibration data.
        self.calibrationData = json.load(open('./data/calibrate_data.json', 'r'))
        self.pixelsPerMetric = self.calibrationData['pixel-per-metric']
        print("[INFO] Calibration data successfully loaded.")

        if operation == 'train':
            self.train_system(image)
            return

        if os.path.isfile('data/reference_data.json') == False:
            print("\n[ERROR] Reference data not found. Please train the system first.\n")
            return False

        # Load the reference data.
        self.referenceData = json.load(open('./data/reference_data.json', 'r'))
        print("[INFO] Reference data successfully loaded.")

        if operation == 'test':
            self.test_part(image)
    
    def get_system_data(self, log=True):
        if log:
            print("[INFO] Pixel-per-metric ratio is {}\n".format(self.pixelsPerMetric))
            
            print("[INFO] Reference data is: \n")
            for data in self.referenceData:
                print("ID: {}".format(data['id']))
                print("Area: {}".format(data['area']))
                print("Center: {}".format(data['center']))
                print("Dimensions: {}\n".format(data['dimensions']))

            print("[INFO] Showing reference image.\n")
            cv2.imshow("image", self.referenceImage)
            cv2.waitKey(0)

        return self.calibrationData, self.referenceData, self.referenceImage

    def calibrate_system(self, calibrate_image):
        print("[INFO] Calibration started!")

        # Preprocess the reference image.
        preprocessed_image = preprocess_image(calibrate_image, show=True)[0]

        # Get contours from the preprocessed image.
        contours = get_contours_from_image(preprocessed_image)

        # Ensure there's a single reference object present.
        assert len(contours) > 0, "[ERROR] Invalid reference object, Please try again..."
        calibrate_object = max(contours, key=cv2.contourArea)

        # Calculate the pixel-per-metric ratio.
        self.pixelsPerMetric = analyze_contour(calibrate_object, calibrate=True) / self.REFERENCE_WIDTH
        print("[INFO] Calibration completed, pixel-per-metric ratio is {}\n".format(self.pixelsPerMetric))

        # Save the calibration data.
        file = open('data/calibrate_data.json', 'w+')
        data = {'pixel-per-metric': self.pixelsPerMetric}
        json.dump(data, file)
    
    def train_system(self, reference_image):
        print("[INFO] Training started!")

        # Compute the reference data & get the reference image.
        self.referenceData, self.referenceImage = get_features(reference_image, self.pixelsPerMetric)

        # Save the reference images.
        cv2.imwrite('data/reference_image.jpg', self.referenceImage)

        # Save the reference data.
        file = open('data/reference_data.json', 'w+')
        json.dump(self.referenceData, file)

        print("[INFO] Training successful.")
        print("[INFO] Reference data successfully saved.")
    

    def test_part(self, part_image):
        print("[INFO] Analyzing part.")
        # Compute the part data & get the part image.
        partData, partImage = get_features(part_image, self.pixelsPerMetric)
        feature_diff = compare_features(partData, self.referenceData)

        for i in feature_diff:
            print(i, "\n")
