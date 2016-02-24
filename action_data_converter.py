import os
import cv2
import copy
import utils as u
import numpy as np
import itertools
from progressbar import ProgressBar

# Usage: The videos are saved in '/om/data/public/mbchang/udcign-data/action/raw/videos'
# I haven't figured out how to use cv2 on openmind yet, so copy the videos
# to your local computer as the root folder below. There should be a folder for
# each of the actions below under the videos folder

# pc
root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/data/udcign/action/videos'
out = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/data/udcign/action/hdf5'

# openmind
# root = '/om/data/public/mbchang/udcign-data/action/raw/videos'
# out = '/om/data/public/mbchang/udcign-data/action/raw/hdf5'

actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
scenario = 'd4'  # d4 means outdoors
subsample = 1
gray = True

action_data = {}

for action in actions:
    print action
    action_vid_folder = os.path.join(root,action)
    action_vids = {}
    pbar = ProgressBar()
    for i in pbar(range(len(os.listdir(action_vid_folder)))):
        vid = os.listdir(action_vid_folder)[i]
        if scenario not in vid: continue
        # print vid
        # continue
        cap = cv2.VideoCapture(os.path.join(action_vid_folder, vid))
        num_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        video = []
        # pbar = ProgressBar()

        # the frames are guaranteed to be consecutive
        for i in range(int(num_frames)):
            ret, frame = cap.read()
            # print frame

            if gray: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # shape: (height, width)  it is gray anyway
            if i % subsample == 0:
                frame = frame/float(255)  # normalize
                frame = frame.astype('float32')
                # cv2.imshow('frame',frame)
                if gray: frame = np.tile(frame,(1,1,1))  # give it the channel dim
                video.append(frame)

                # import time
                # time.sleep(0.5)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        # cv2.destroyAllWindows()

        video = u.stack(video)
        action_vids[vid] = video  # video, subsampled, evenly spaced, consecutive video
    pbar.finish()
    u.save_dict_to_hdf5(dataset=action_vids, dataset_name=action+'_subsamp='+str(subsample)+'_scenario='+scenario, dataset_folder=out)

    # action_vids = np.vstack(action_vids)  # consecutive video  ACTUALLY THIS MIGHT NOT BE TRUE. WE NEED THE VIDEOS TO BE SEPARATE!

    # randomly permute  -- don't do this!
    # tm1s = np.random.permutation(range(0,len(action_vids)-1,2))
    # ts = np.array([i+1 for i in tm1s])
    # shuffle_idxs = list(it.next() for it in itertools.cycle([iter(tm1s), iter(ts)])) # groups of 2
    # action_vids = action_vids[np.array(shuffle_idxs),:,:]

    # action_data[action] = action_vids

# save
# u.save_dict_to_hdf5(action_data, 'actions_2_frame_subsample_' + str(subsample), root)
