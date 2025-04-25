import numpy as np
import pickle
import h5py
import os.path as osp
import glob

from mappings import activity_mappings

def reformat_pkl_data(samples):
    formatted_data = []

    for sample in samples:
        with open(sample, "rb") as f:
            activity_data = pickle.load(f)

        frames = []
        for frame in activity_data:
            frames.append(np.array(frame))
        frames = np.array(frames)

        formatted_data.append(frames)

    # flatten from (n, 25, 3) to (n, 75)
    formatted_data = [x.reshape(-1, 75) for x in formatted_data]

    return np.array(formatted_data, dtype=object)


def align_frames(skel_data):
    max_num_frames = max([x.shape[0] for x in skel_data])
    aligned_frames = np.zeros((len(skel_data), max_num_frames, 75), dtype=np.float32)

    for idx, frame in enumerate(skel_data):
        num_frames = frame.shape[0]
        aligned_frames[idx, :num_frames] = frame

    return aligned_frames

def one_hot_vector(labels, m="a"):
    if m == "a":
        n = 2
    elif m == "p":
        n = 8

    one_hot = np.zeros((len(labels), n))
    for i, label in enumerate(labels):
        one_hot[i, label-1] = 1

    return one_hot

def get_data_and_labels(aligned_data, sequences):
    activities = []
    activity_names = []
    performers = []
    for seq in sequences:
        person_id = int(seq.split("/")[-4][6:])
        activity = seq.split("/")[-3][3:]
        activity_id = activity_mappings[activity]
        activities.append(activity_id)
        activity_names.append(activity)
        performers.append(person_id)

    # re-index labels to start from 0
    activity_mapping = {activity: idx for idx, activity in enumerate(np.unique(activities))}
    activities = np.array([activity_mapping[activity] for activity in activities])
    performer_names = performers
    performer_mapping = {performer: idx for idx, performer in enumerate(np.unique(performers))}
    performers = np.array([performer_mapping[performer] for performer in performers])

    test_data = {}
    test_data["data"] = aligned_data
    test_data["activities"] =  activities
    test_data["activity_names"] = activity_names
    test_data["performers"] = performers
    test_data["performer_names"] = performer_names

    return test_data


def save_dataset(test_data):
    with h5py.File("/u/home/aoma/gr/SGN/data/sitc/PP-SITC_TC2-1_onlyTest.h5", 'w') as f:
        f.create_group("test")

        test_group = f["test"]
        test_group.create_dataset("x", data=test_data["data"])
        print(f"Test data shape: {test_data['data'].shape}")
        test_group.create_dataset("a", data=one_hot_vector(test_data["activities"], m="a"))
        test_group.create_dataset("p", data=one_hot_vector(test_data["performers"], m="p"))
        test_group.create_dataset("activity_names", data=test_data["activity_names"])
        test_group.create_dataset("performer_names", data=test_data["performer_names"])



if __name__ == '__main__':
    only_test = True

    retargeted_dir = "/u/home/aoma/gr/datasets/SITC/retargeted_data"
    sequence_files = glob.glob(osp.join(retargeted_dir, "**", "results.pkl"), recursive=True)

    with open("/u/home/aoma/gr/datasets/SITC/statistics/test_samples.txt", "r") as f:
        sequences_to_use = f.readlines()

    sequences_to_use = [x.strip() for x in sequences_to_use]

    if only_test:
        sequences = []
        for x in sequence_files:
            person_id = x.split("/")[-4][6:]
            activity = x.split("/")[-3][3:]
            activity_id = activity_mappings[activity]
            repl_nr = x.split("/")[-2]

            # for each sequence in sequences_to_use, match the pattern P{person_id}A{activity_id}C{any number}R{repl_nr}
            for seq in sequences_to_use:
                seq = seq.strip()
                if f"P{person_id}A{str(activity_id).zfill(3)}C" in seq and f"R{repl_nr.zfill(3)}" in seq:
                    print(f"Added sequence {seq}")
                    sequences.append(x)
                    break
    else:
        sequences = sequence_files

    print(f"Found {len(sequences)} sequences")

    reformatted_data = reformat_pkl_data(sequences)
    print(f"Reformatted data shape: {reformatted_data.shape}")
    aligned_data = align_frames(reformatted_data)
    print(f"Aligned data shape: {aligned_data.shape}")

    test_data = get_data_and_labels(aligned_data, sequences)
    save_dataset(test_data)
    print("Done!")