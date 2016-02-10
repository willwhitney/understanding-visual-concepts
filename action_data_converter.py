import os
import cv2
import copy
import utils as u
import numpy as np
import itertools

root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/unsupervised-dcign/data/actions/raw/videos'
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
subsample = 5

action_data = {}

for action in actions[:2]:
    action_vid_folder = os.path.join(root,action)
    action_vids = []
    for vid in os.listdir(action_vid_folder)[:2]:
        cap = cv2.VideoCapture(os.path.join(action_vid_folder, vid))
        num_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        video = []

        i = 0
        while(i < num_frames):
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # shape: (height, width)
            if i % subsample == 0:
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
        print(video.shape)

    action_vids = np.vstack(action_vids)  # odd idxes are tm1, even are t

    # randomly permute
    tm1s = np.random.permutation(range(0,len(action_vids)-1,2))
    ts = np.array([i+1 for i in tm1s])
    shuffle_idxs = list(it.next() for it in itertools.cycle([iter(tm1s), iter(ts)]))
    action_vids = action_vids[np.array(shuffle_idxs),:,:]

    action_data[action] = action_vids

# save
u.save_dict_to_hdf5(action_data, 'test_actions_2_frame_subsample_' + str(subsample), root)
