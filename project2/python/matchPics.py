import cv2
import numpy as np

def matchPics(I1:np.ndarray, I2:np.ndarray):
    """ match FAST features between two images

    Args:
        I1 (_type_): image 1
        I2 (_type_): image 2

    Returns:
        tuple(locs1, locs2): matched keypoints on image 1 and image 2
    """
#MATCHPICS Extract features, obtain their descriptors, and match them!

## Convert images to grayscale, if necessary
    if I1.ndim == 3:
        img1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    if I2.ndim == 3:
        img2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

## Detect features in both images
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    # Print all  params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints on Image1: {}".format(len(kp1)) )
    print( "Total Keypoints on Image2: {}".format(len(kp2)) )

## Obtain descriptors for the computed feature locations
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)
    # Print all the params of BRIEF
    print( "Descriptor Size: {}".format(brief.descriptorSize()) )
    print( "Bytes: {}".format(brief.descriptorSize()) )
    print( "Total descriptors on Image1: {}".format(des1.shape) )
    print( "Total descriptors on Image2: {}".format(des2.shape) )

## Match features using the descriptors
    mathcer = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = mathcer.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good_matches = []
    matcher_ratio = 0.75
    for m, n in matches:
        if m.distance < matcher_ratio*n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x:x.distance)
    print( "Total matches: {}".format(len(good_matches)) )

    # return locs1, locs2
    locs1 = np.array([kp1[mat.queryIdx].pt for mat in good_matches])
    locs2 = np.array([kp2[mat.trainIdx].pt for mat in good_matches])
    
    return locs1, locs2

if __name__=="__main__":
    # Load images
    cv_img = cv2.imread('../data/cv_cover.jpg')
    desk_img = cv2.imread('../data/cv_desk.png')

    # Extract features and match
    locs1, locs2 = matchPics(cv_img, desk_img)

    # vis matches
    num_kpts = 100
    kpts1 = locs1[:num_kpts]
    kpts2 = locs2[:num_kpts]
    kp1 = [cv2.KeyPoint(x=kpt[0], y=kpt[1], size=1) for kpt in kpts1]
    kp2 = [cv2.KeyPoint(x=kpt[0], y=kpt[1], size=1) for kpt in kpts2]
    matches = [cv2.DMatch(i, i, 0) for i in range(num_kpts)] 
    match_result = cv2.drawMatches(cv_img, kp1, desk_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('./results/matchPics.jpg', match_result)
    cv2.imshow('show all matchings', match_result)
    cv2.waitKey()
