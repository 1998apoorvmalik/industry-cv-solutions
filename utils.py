import numpy as np
import imutils
from imutils import contours
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
import random

# Specified colors to draw the distances along with the reference object.
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))

def midpoint(ptA, ptB):
    return(((ptA[0] + ptB[0])/2), ((ptA[1] + ptB[1])/2))

def preprocess_image(image, show=True):
    print("[INFO] Pre-processing the image.")
    
    # Convert the image to grayscale, and blur it slightly.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Thresholding.
    thresh_1 = cv2.threshold(gray, 150, 200, cv2.THRESH_BINARY_INV)[1]
    thresh_2 = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)[1]

    sigma=0.8
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Process for image type 1.
    img_1 = cv2.Canny(gray, lower, upper)                        # Edge Detection
    img_1 = cv2.dilate(img_1, None, iterations=1)               # Dialation
    img_1 = cv2.erode(img_1, None, iterations=1)                # Erosion

    # Process for image type 2.
    img_2 = cv2.Canny(thresh_2, lower, upper)                        # Edge Detection 
    img_2 = cv2.dilate(img_2, None, iterations=1)               # Dialation
    img_2 = cv2.erode(img_2, None, iterations=1)                # Erosion

    # Show the preprocess image if log == True.
    if show:
        log_image = np.concatenate((cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR),\
            cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)), axis=1)
        cv2.imshow("image", log_image)
        cv2.waitKey(0)

    return [img_1, img_2]

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def get_contours_from_image(image):
    print("[INFO] Finding contours.")
    # Find contours in image.
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = imutils.grab_contours(cnts)

    # Sort the contours from top-to-bottom.
    # cnts.sort(key=lambda x:get_contour_precedence(x, image.shape[1]))
    # (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
    # sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])
    return list(cnts)

def get_features(image, pixelsPerMetric, min_size=500, show=True):
    # Initialize an array to store the features of the object.
    features = []

    # Preprocess the input image.
    reference_image, features_image = preprocess_image(image, show=True)

    # Get the reference object contour
    reference_contour = max(get_contours_from_image(reference_image), key=cv2.contourArea)

    # Find contours in the features image.
    feature_contours = get_contours_from_image(features_image)

    # Copy the background image.
    display_image = image.copy()

    # Loop over the feature contours individually.
    for (i, contour) in enumerate([reference_contour] + feature_contours):
        # Analyze contour and extract its features.
        if i > 0:
            feature = analyze_contour(contour, pixelsPerMetric, features[0])
        else:
            feature = analyze_contour(contour, pixelsPerMetric)

        # If the contour is not sufficiently large, ignore it.
        if feature['area'] < min_size:
            continue

        # Show contours info over the background image.
        if i > 0:
            display_image = draw_feature(display_image, feature, features[0])
        else:
            display_image = draw_feature(display_image, feature)
        
        features.append(feature)

        if show:
            # Show the output image.
            cv2.imshow("image", display_image)
            cv2.waitKey(0)

    return features, display_image

def analyze_contour(contour, pixelsPerMetric=None, reference_feature=None, calibrate=False):
    # Object containing the features of the contour.
    feature = dict()

    # Compute the area of the contour.
    area = cv2.contourArea(contour)

    # Compute the rotated bounding box of the contour.
    box = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="float")

    # Order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order.
    box = perspective.order_points(box)

    # Compute the center of the bounding box.
    cX = float(np.average(box[:, 0]))
    cY = float(np.average(box[:, 1]))

    # Unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates.
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # Compute the midpoint between the top-left and bottom-left points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # Calibrate argument check.
    if calibrate:
        return dB

    # Ensure that pixel-per-metric value is supplied.
    assert pixelsPerMetric != None, "Pixel per metric value not supplied."

    # Compute the dimensions of the object.
    height = dA / pixelsPerMetric
    width = dB / pixelsPerMetric

    # Adding values to the feature dictionary.
    feature['contour'] = contour.tolist()
    feature['box'] = box.tolist()
    feature['area'] = area
    feature['center'] = (cX, cY)
    feature['dimensions'] = (height, width)
    feature['midpoints'] = [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]

    if reference_feature != None:
        feature['reference_distance'] = analyze_feature_reference_gap(feature, reference_feature, pixelsPerMetric)

    return feature

def analyze_feature_reference_gap(current_feature, reference_feature, pixelsPerMetric):
    # Stack the current feature coordinates and the reference feature coordinates
	# to include the feature's center.
    currentFeatureCoords = np.vstack([current_feature['box'], current_feature['center']])
    referenceFeatureCoords = np.vstack([reference_feature['box'], reference_feature['center']])

    # Array to store distances between the various edges & center of the two features.
    distances = []

    # Loop over the original points.
    for ((xA, yA), (xB, yB), color) in zip(referenceFeatureCoords, currentFeatureCoords, colors):
		# Compute the Euclidean distance between the coordinates,
		# and then convert the distance in pixels to distance in
		# units.
        D = dist.euclidean((xA, yA), (xB, yB)) / pixelsPerMetric
        distances.append(D)
    
    return distances
    
def compare_features(feature_1, feature_2):
    feature_diff_array = []

    for i in range(len(feature_1)):
        feature_diff = dict()
        feature_diff['dimensions'] = abs(np.array(feature_1[i]['dimensions']) - np.array(feature_2[i]['dimensions'][0]))
        feature_diff['area'] = abs(feature_1[i]['area'] - feature_2[i]['area'])
        feature_diff['center'] = abs(np.array(feature_1[i]['center']) - np.array(feature_2[i]['center']))

        if i > 0:
            feature_diff['reference_distance'] = abs(np.array(feature_1[i]['reference_distance']) -\
                np.array(feature_2[i]['reference_distance']))
        
        feature_diff_array.append(feature_diff)
    
    return feature_diff_array

def draw_feature(image, current_feature, reference_feature=None):
    # Loop over the original points and draw them.
    for (x, y) in current_feature['box']:
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    # Draw the original contour.
    cv2.drawContours(image, np.array(current_feature['contour']), -1, (255, 255, 255), 3)

    # Draw the outline of the rotated bounding box.
    cv2.drawContours(image, [np.array(current_feature['box']).astype("int")], -1, (255, 150, 100), 2)

    # Get midpoints.
    (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY) = current_feature['midpoints']

    # Draw the midpoints on the image.
    cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # Draw lines between the midpoints.
    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    # Object dimensions.
    dimA, dimB = current_feature['dimensions'] 

    # Draw contour & info on the image.
    cv2.putText(image, "{:.1f}mm".format(dimA),
        (int(blbrX + 15), int(blbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2)
    cv2.putText(image, "{:.1f}mm".format(dimB),
        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2)
    cv2.putText(image, "Area: {}".format(current_feature['area']),
        (int(tlblX - 75), int(tlblY + 75)), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2)

    if reference_feature != None:
        # Stack the current feature coordinates and the reference feature coordinates
        # to include the feature's center.
        currentFeatureCoords = np.vstack([current_feature['box'], current_feature['center']])
        referenceFeatureCoords = np.vstack([reference_feature['box'], reference_feature['center']])

        # loop over the original points
        for ((xA, yA), (xB, yB), D, color) in zip(referenceFeatureCoords, currentFeatureCoords,\
            current_feature['reference_distance'], colors):
            # draw circles corresponding to the current points and
            # connect them with a line
            cv2.circle(image, (int(xA), int(yA)), 5, color, -1)
            cv2.circle(image, (int(xB), int(yB)), 5, color, -1)
            cv2.line(image, (int(xA), int(yA)), (int(xB), int(yB)),
                color, 2)

            (mX, mY) = midpoint((xA, yA), (xB, yB))
            cv2.putText(image, "{:.1f}mm".format(D), (int(mX), int(mY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    
    return image