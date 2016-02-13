import os
import cv2
import copy
import utils as u
import numpy as np
import itertools

# Usage: The videos are saved in '/om/data/public/mbchang/udcign-data/action/raw/videos'
# I haven't figured out how to use cv2 on openmind yet, so copy the videos
# to your local computer as the root folder below. There should be a folder for
# each of the actions below under the videos folder

root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/data/udcign/action/videos'
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
subsample = 5

action_data = {}

for action in actions:
    action_vid_folder = os.path.join(root,action)
    action_vids = []
    for vid in os.listdir(action_vid_folder):
        cap = cv2.VideoCapture(os.path.join(action_vid_folder, vid))
        num_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        video = []

        i = 0
        while(i < num_frames):
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # shape: (height, width)
            if i % subsample == 0:
                gray = gray/float(255)  # normalize
                gray = gray.astype('float32')

                video.append(gray)

                # cv2.imshow('frame',gray)
                # import time
                # time.sleep(0.5)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1

        cap.release()
        cv2.destroyAllWindows()

        video = u.stack(video)
        if video.shape[0] % 2 != 0: video = video[:-1,:,:]
        action_vids.append(video)  # video, subsampled, evenly spaced

    action_vids = np.vstack(action_vids)  # consecutive video

    # randomly permute  -- don't do this!
    # tm1s = np.random.permutation(range(0,len(action_vids)-1,2))
    # ts = np.array([i+1 for i in tm1s])
    # shuffle_idxs = list(it.next() for it in itertools.cycle([iter(tm1s), iter(ts)])) # groups of 2
    # action_vids = action_vids[np.array(shuffle_idxs),:,:]

    action_data[action] = action_vids

# save
u.save_dict_to_hdf5(action_data, 'actions_2_frame_subsample_' + str(subsample), root)
