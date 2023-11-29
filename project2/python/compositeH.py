import cv2
import numpy as np

def compositeH(H2to1, template, img):
    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1 * x_photo
    # For warping the template to the image, we need to invert it.
    H_template_to_img = np.linalg.inv(H2to1)

    # Create mask of same size as template
    mask = np.ones(template.shape)

    # Warp mask by appropriate homography
    mask = cv2.warpPerspective(mask, H_template_to_img, (img.shape[1], img.shape[0]))

    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H_template_to_img, (img.shape[1], img.shape[0]))

    # Use mask to combine the warped template and the image
    composite_img = img * (1-mask) + warped_template * mask
    return composite_img