# Q2.7
import os
import sys
sys.path.append('./')
sys.path.append('../')

import cv2
import numpy as np

from loadVid import loadVid
from matchPics import matchPics
from computeH_ransac import computeH_ransac
from warpH import warpH
from compositeH import compositeH

def get_aligned_frames_pair(video1, video2):
    video1_frames_lst = []
    video2_frames_lst = []
    for frame1_dict, frame2_dict in zip(video1, video2):
        if frame1_dict is not None and frame2_dict is not None:
            frame1 = frame1_dict['cdata']
            frame2 = frame2_dict['cdata']

            img_w_1, img_h_1 = frame1.shape[1], frame1.shape[0]
            img_w_2, img_h_2 = frame2.shape[1], frame2.shape[0]
            # width = min(img_w_1, img_w_2)
            # height = min(img_h_1, img_h_2)
            # take video2 as reference
            width, height = img_w_2, img_h_2
            asp_ratio = width / height
            # crop the video1 frame
            if img_w_1 / img_h_1 > asp_ratio:
                new_w = int(asp_ratio * img_h_1)
                frame1 = frame1[:, (img_w_1 - new_w) // 2: (img_w_1 - new_w) // 2 + new_w]
            else:
                new_h = int(img_w_1 / asp_ratio)
                frame1 = frame1[(img_h_1 - new_h) // 2: (img_h_1 - new_h) // 2 + new_h, :]

            # frame1 = cv2.resize(frame1, (width, height))
            # frame2 = cv2.resize(frame2, (width, height))
            video1_frames_lst.append(frame1)
            video2_frames_lst.append(frame2)
    return video1_frames_lst, video2_frames_lst

def create_video_from_frames(frames_lst, video_name):
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (frames_lst[0].shape[1], frames_lst[0].shape[0]))

    for frame in frames_lst:
        out.write(frame)
    out.release()



if __name__=="__main__":
    video1_path = '../data/ar_source.mov'
    video2_path = '../data/book.mov'
    video_kungfu = loadVid(video1_path)
    video_mvg = loadVid(video2_path)

    img_mvg_cover = cv2.imread('../data/cv_cover.jpg')

    # get aligned frames
    video_kf_frames_lst, video_mvg_frames_lst = get_aligned_frames_pair(video_kungfu, video_mvg)

    # get homography for each frame pair
    homography_lst = []
    composite_img_lst = []
    for i, (frame1, frame2) in enumerate(zip(video_kf_frames_lst, video_mvg_frames_lst)):
        # compute homography between book cover and frame2
        locs1, locs2 = matchPics(img_mvg_cover, frame2)
        bestH_frame2tocover, _ = computeH_ransac(locs1, locs2)
        homography_lst.append(bestH_frame2tocover)

        # warp frame1 to frame2
        H1to2 = np.linalg.inv(bestH_frame2tocover)
        frame1 = cv2.resize(frame1, (img_mvg_cover.shape[1], img_mvg_cover.shape[0]))
        warped_img = warpH(frame1, H1to2, frame2.shape)
        # cv2.imwrite('./results/warped_hp.jpg', warped_img)

        # Display composite image
        composite_img = compositeH(bestH_frame2tocover, frame1, frame2)
        cv2.imwrite(f'./results/ar/composite_img_{i}.jpg', composite_img)
        print(f'composed image data type: {composite_img.dtype}')
        composite_img = composite_img.astype(np.uint8)
        composite_img_lst.append(composite_img)

    # save video
    video_name = './results/ar.avi'
    create_video_from_frames(composite_img_lst, video_name)



    


