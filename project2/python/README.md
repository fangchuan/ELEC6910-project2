# PROJECT2: Planar Homographics

## 1. Task 2.1

Run `python matchPics.py` will give you visualization of feature matches between cv_cover.png and cv_desk.png, you can check the first 100 matches by results in `./results/matchPics.png`.

## 2. Task 2.2

Run `python briefRotTest.py` will give you different number of feasture matches between each the rotated image and the original image. You can find the visualized feature matches: `./results/brief_rotation_30.png`, `./results/brief_rotation_150.png`, and `./results/brief_rotation_300.png` under 30, 150, and 300 degree rotation respectively. And the feauture matches histogram: `./results/brief_rot_hist.png`. We found that BRIEF features work well under small rotation, but not good under large rotation. The reason is that BRIEF features are binary features, which are not rotation invariant, and it only considers the intensity of the pixels in the patch, which is not robust to rotation.

## 3. Task 2.3
Run `python computeH.py` will give you the desired the homograpy matrix between cv_cover.png and cv_desk.png. You can find the validation images in `./results/compute_H_matches.png` and `./results/compute_H_warped_img.png` .

## 4. Task 2.4
Run `python computeH_norm.py` will give you the desired the homograpy matrix between cv_cover.png and cv_desk.png. You can find the validation images in `./results/compute_H_norm_matches.png` and `./results/compute_H_norm_warped_img.png` .


## 5. Task 2.5
Run `python computeH_ransac.py` will give you the desired the homograpy matrix between cv_cover.png and cv_desk.png. You can find the validation images in `./results/compute_H_ransac_matches.png` and `./results/compute_H_ransac_warped_img.png` . We further adopt the iterative RANSAC algorithm to speed up the computation. 

## 6. Task 2.6
Run `python HarryPotterize_auto.py` will give you the desired warpped image in `./results/HP_DESK.png` . 

## 7. Task 2.7
Run `python ar.py` will give you the desired blended images under folder `./results/ar/` and the final vedieo `./results/ar.avi`.
