
import cv2
import numpy as np

from scipy import ndimage

FEATURE_TYPES = ['sift', 'surf', 'orb', 'brisk', 'brief']

feature_method = 'brief'

# Read the image and convert to grayscale if necessary
img_filepath = '../data/cv_cover.jpg'
img = cv2.imread(img_filepath)
if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Create a histogram to store the results
kpt_matches_lst = []

# Compute the features and descriptors for the original image
fast_detector = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
ori_kpts = fast_detector.detect(img, None)
if feature_method == 'brief':
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    ori_kpt, ori_des = brief.compute(img, ori_kpts)
    print(f'Orighinal image detect {len(ori_kpt)} keypoints, BRIEF desciptor size: {ori_des.shape}')
elif feature_method == 'surf':
    surf = cv2.xfeatures2d.SURF_create()
    ori_kpt, ori_des = surf.compute(img, ori_kpts)
    print(f'Orighinal image detect {len(ori_kpt)} keypoints, SURF desciptor size: {ori_des.shape}')
elif feature_method == 'sift':
    sift = cv2.xfeatures2d.SIFT_create()
    ori_kpt, ori_des = sift.detectAndCompute(img, None)
    print(f'Orighinal image detect {len(ori_kpt)} keypoints, SIFT desciptor size: {ori_des.shape}')

mathcer = cv2.BFMatcher(cv2.NORM_HAMMING)

matcher_ratio = 0.75
for i in range(37):
    # Rotate the image
    roted_img = ndimage.rotate(img, i*10)

    # Compute features and descriptors for the rotated image
    roted_kpts = fast_detector.detect(roted_img, None)
    if feature_method == 'brief':
        roted_kpt, roted_des = brief.compute(roted_img, roted_kpts)
        print(f'Rotated image detect {len(roted_kpt)} keypoints, BRIEF desciptor size: {roted_des.shape}')
    elif feature_method == 'surf':
        roted_kpt, roted_des = surf.compute(roted_img, roted_kpts)
        print(f'Rotated image detect {len(roted_kpt)} keypoints, SURF desciptor size: {roted_des.shape}')

    # Match features
    matches = mathcer.knnMatch(ori_des, roted_des, k=2)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < matcher_ratio*n.distance:
            good_matches.append(m)
    print( "Image {} matches: {}".format(i, len(good_matches)) )

    # save the matched image
    if i in [3, 15, 30]:
        match_result = cv2.drawMatches(img, ori_kpt, roted_img, roted_kpt, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f'./results/{feature_method}_rotation_{i*10}.jpg', match_result)
    # Update histogram
    kpt_matches_lst.append(len(good_matches))


# Display histogram
import matplotlib.pyplot as plt
plt.bar(range(37), kpt_matches_lst)
plt.xlabel('Rotation angle (degree)')
plt.ylabel('Number of matched keypoints')
plt.title('Number of matched keypoints vs rotation angle')
plt.show()
