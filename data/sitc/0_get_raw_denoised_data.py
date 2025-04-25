import os.path as osp
import numpy as np
import pandas as pd
import pickle
import json
import os

from mappings import kitchen_sensors, activity_names, activity_mappings, coco2NTU, subsequences_csv, base_folder

def convert_keypoints(frame_kpts, origin=None, with_BODY25=False):
    new_frame_kpts = np.zeros((25,3))

    for key in coco2NTU:
        new_frame_kpts[coco2NTU[key]-1] = frame_kpts[key]

    new_frame_kpts[20] = [(frame_kpts[5][0] + frame_kpts[6][0]) / 2, (frame_kpts[5][1] + frame_kpts[6][1]) / 2, (frame_kpts[5][2] + frame_kpts[6][2]) / 2]
    new_frame_kpts[0] = [(frame_kpts[11][0] + frame_kpts[12][0]) / 2, (frame_kpts[11][1] + frame_kpts[12][1]) / 2, (frame_kpts[11][2] + frame_kpts[12][2]) / 2]
    new_frame_kpts[15] = frame_kpts[14]
    new_frame_kpts[19] = frame_kpts[11]
    new_frame_kpts[1] = (new_frame_kpts[20] + new_frame_kpts[0]) / 2
    new_frame_kpts[2] = (2 * new_frame_kpts[20] + new_frame_kpts[3]) / 3
    new_frame_kpts[7] = new_frame_kpts[6]
    new_frame_kpts[21] = new_frame_kpts[6]
    new_frame_kpts[22] = new_frame_kpts[6]
    new_frame_kpts[11] = new_frame_kpts[10]
    new_frame_kpts[23] = new_frame_kpts[10]
    new_frame_kpts[24] = new_frame_kpts[10]
    new_frame_kpts[:,2] = 0 # set z coordinate to 0

    if origin is None:
        origin = np.copy(new_frame_kpts[1])

    new_frame_kpts = new_frame_kpts - np.tile(origin, (25,1))
    new_frame_kpts = np.array([-new_frame_kpts[:,1], -new_frame_kpts[:,0], new_frame_kpts[:,2]]).T # rotate the skeleton to match NTU dataset  
    new_frame_kpts = new_frame_kpts.reshape(-1) # concatenate each 3D coord along the row dimension in joint order (25x3 -> 75)
    
    if with_BODY25:
        new_frame_body25 = {}
        new_frame_body25["version"] = 1.2
        new_frame_body25["people"] = []
        
        new_frame_kpts_body25 = {}
        std_const = frame_kpts[4]
        std_const[2] = 0

        new_frame_kpts_body25[0] = np.array(frame_kpts[0]) - np.array(std_const)
        new_frame_kpts_body25[1] = np.array([(frame_kpts[5][0] + frame_kpts[6][0]) / 2, (frame_kpts[5][1] + frame_kpts[6][1]) / 2, (frame_kpts[5][2] + frame_kpts[6][2]) / 2]) - np.array(std_const)
        new_frame_kpts_body25[2] = np.array(frame_kpts[6]) - np.array(std_const)
        new_frame_kpts_body25[3] = np.array(frame_kpts[8]) - np.array(std_const)
        new_frame_kpts_body25[4] = np.array(frame_kpts[10]) - np.array(std_const)
        new_frame_kpts_body25[5] = np.array(frame_kpts[5]) - np.array(std_const)
        new_frame_kpts_body25[6] = np.array(frame_kpts[7]) - np.array(std_const)
        new_frame_kpts_body25[7] = np.array(frame_kpts[9]) - np.array(std_const)
        new_frame_kpts_body25[8] = np.array([(frame_kpts[11][0] + frame_kpts[12][0]) / 2, (frame_kpts[11][1] + frame_kpts[12][1]) / 2, (frame_kpts[11][2] + frame_kpts[12][2]) / 2]) - np.array(std_const)
        new_frame_kpts_body25[9] = np.array(frame_kpts[12]) - np.array(std_const)
        new_frame_kpts_body25[10] = np.array(frame_kpts[14]) - np.array(std_const)
        new_frame_kpts_body25[11] = np.array(frame_kpts[16]) - np.array(std_const)
        new_frame_kpts_body25[12] = np.array(frame_kpts[11]) - np.array(std_const)
        new_frame_kpts_body25[13] = np.array(frame_kpts[13]) - np.array(std_const)
        new_frame_kpts_body25[14] = np.array(frame_kpts[15]) - np.array(std_const)
        new_frame_kpts_body25[15] = np.array(frame_kpts[2]) - np.array(std_const)
        new_frame_kpts_body25[16] = np.array(frame_kpts[1]) - np.array(std_const)
        new_frame_kpts_body25[17] = np.array(frame_kpts[4]) - np.array(std_const)
        new_frame_kpts_body25[18] = np.array(frame_kpts[3]) - np.array(std_const)

        new_frame_kpts_body25[19] = new_frame_kpts_body25[14]
        new_frame_kpts_body25[20] = new_frame_kpts_body25[14]
        new_frame_kpts_body25[21] = new_frame_kpts_body25[14]
        new_frame_kpts_body25[22] = new_frame_kpts_body25[11]
        new_frame_kpts_body25[23] = new_frame_kpts_body25[11]
        new_frame_kpts_body25[24] = new_frame_kpts_body25[11]

        new_frame_kpts_body25 = [item for sublist in new_frame_kpts_body25.values() for item in sublist]
        
        new_frame_body25["people"].append({"pose_keypoints_2d": new_frame_kpts_body25, "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": [], 
                                    "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": []})
        
        return new_frame_kpts, origin, new_frame_body25
    
    return new_frame_kpts, origin

def get_splits(kpts_interval, confidence_interval=0.5):
    splits = []
    split = []

    for k, v in kpts_interval.items():
        if np.any(np.array(v)[:, 2] < confidence_interval): 
            splits.append(split)
            split = []
            continue
        split.append(k) # only append a frame if all keypoints are above the confidence threshold
    splits.append(split)

    return splits

def get_df(path):
    df = pd.read_csv(path, delimiter=';', header=None)
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(0)
    return df

def get_activity_id(activity):
    if activity.startswith("_"):
        activity_id = activity_mappings[activity[1:]]
        activity = f"K{kitchen_sensors[i]}{activity}"
    else:
        activity_id = activity_mappings[activity[3:]]

    return activity_id, activity

if __name__ == "__main__":
    df = get_df(subsequences_csv)

    for i, col in enumerate(df.columns):
        person_id = i+1
        for row, activity in enumerate(activity_names):
            activity_id, activity = get_activity_id(activity)
            sequence_nrs = json.loads(df.iloc[row][col])

            if sequence_nrs == []:
                continue
            
            else:
                keypoints_file = f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints_interp_spline.json"

                with open(keypoints_file) as f:
                    kpts = json.load(f) # keypoints in COCO format

                interval_file = f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/subsequences.txt"

                with open(interval_file) as f:
                    intervals = f.readlines()

                repl_nr = 1
                new_activity = {}

                for seq_nr in sequence_nrs:
                    print(f"Person {person_id}, activity {activity}, sequence {seq_nr}")
                    interval = intervals[seq_nr].split('\n')
                    interval = [x.split(",") for x in interval[0].split(":")][1][:2]
                    interval = [int(x.strip()) for x in interval]
                    kpts_interval = {k: v for k, v in kpts.items() if interval[0] <= int(k) <= interval[1]}
                    splits = get_splits(kpts_interval, confidence_interval=0.5)

                    for split in splits:
                        if len(split) > 30: # only keep sequences with more than 30 frames
                            with open(osp.join(base_folder, "raw_denoised_data", f"P{str(person_id).zfill(3)}-{activity}.txt"), "a") as f:
                                f.write(f"{repl_nr},{split[0]},{split[-1]}\n")

                            new_activity[repl_nr] = {}

                            for frame_id, frame_nr in enumerate(split):
                                frame_kpts = kpts_interval[frame_nr][0]
                                if frame_id == 0:
                                    new_frame_kpts, origin, new_frame_body25 = convert_keypoints(frame_kpts, with_BODY25=True)
                                else:
                                    new_frame_kpts, _, new_frame_body25 = convert_keypoints(frame_kpts, origin, with_BODY25=True)

                                if not osp.exists(f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/converted_kpts/body25/{repl_nr}"):
                                    os.makedirs(f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/converted_kpts/body25/{repl_nr}")
                                
                                with open(f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/converted_kpts/body25/{repl_nr}/{frame_nr}.json", "w") as f:
                                    json.dump(new_frame_body25, f)

                                new_activity[repl_nr][frame_nr] = new_frame_kpts.tolist()
                                
                            print(f"Person {person_id}, activity {activity}, replication {repl_nr}")
                        repl_nr += 1

                if new_activity != {}:
                    with open(osp.join(base_folder, "raw_denoised_data", f"P{str(person_id).zfill(3)}-{activity}.pkl"), "wb") as f:
                        pickle.dump(new_activity, f)