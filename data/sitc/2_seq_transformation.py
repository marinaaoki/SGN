import numpy as np
import pickle
import h5py
import os.path as osp
from sklearn.model_selection import StratifiedKFold, train_test_split

from mappings import activity_mappings, camera_mappings, data_dir, base_folder, kitchen_sensors

def reformat_pkl_data(samples, activity_mapping):
    formatted_data = []

    for sample in samples:
        person_id = sample[1:4]
        activity_id = sample[5:8]
        activity_name = list(activity_mappings.keys())[list(activity_mappings.values()).index(int(activity_id))]
        # update with the new activity mapping
        activity_id = activity_mapping[activity_id]
        if activity_name.startswith("_"):
            activity_name = f"K{kitchen_sensors[int(person_id)-1]}" + activity_name
        camera_id = sample[9:12]
        camera_name = list(camera_mappings.keys())[list(camera_mappings.values()).index(int(camera_id))]
        repl_id = sample[13:]

        with open(osp.join(data_dir, f"P{person_id}-{camera_name}_{activity_name}.pkl"), "rb") as f:
            activity_data = pickle.load(f)

        frames = []
        for frame in activity_data[int(repl_id)].values():
            frames.append(np.array(frame))
        frames = np.array(frames)

        formatted_data.append(frames)

    return formatted_data


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

def split_train_val_data(aligned_data, train_val_samples, train_val_activities, train_val_performers, n_folds=4):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds = {}

    for fold, (train_indices, val_indices) in enumerate(skf.split(train_val_samples, train_val_activities)):
        train_samples = [train_val_samples[i] for i in train_indices]
        train_activities = [train_val_activities[i] for i in train_indices]
        train_performers = [train_val_performers[i] for i in train_indices]
        val_samples = [train_val_samples[i] for i in val_indices]
        val_activities = [train_val_activities[i] for i in val_indices]
        val_performers = [train_val_performers[i] for i in val_indices]

        folds[fold] = {
            "train": train_indices,
            "train_samples": train_samples,
            "train_activities": train_activities,
            "train_performers": train_performers,
            "val": val_indices,
            "val_samples": val_samples,
            "val_activities": val_activities,
            "val_performers": val_performers
        }

        train_data = {}
        val_data = {}

        for fold in folds.keys():
            train_data[fold] = {}
            val_data[fold] = {}

            train_samples = folds[fold]["train"]
            val_samples = folds[fold]["val"]

            train_data[fold] = aligned_data[train_samples]
            val_data[fold] = aligned_data[val_samples]

    return train_data, val_data, folds


def split_data(aligned_data, sequences, activities, performers, n_folds=4):
    train_val_indices, test_indices = train_test_split(np.arange(len(sequences)), test_size=0.15, stratify=performers, random_state=42)

    train_val_samples = [sequences[i] for i in train_val_indices]
    # dump in txt file
    with open(osp.join(base_folder, "statistics", "train_val_samples.txt"), "w") as f:
        for sample in train_val_samples:
            f.write(f"{sample}\n")
    train_val_activities = [int(activities[i]) for i in train_val_indices]
    train_val_performers = [int(performers[i]) for i in train_val_indices]

    #test_samples = [sequences[i] for i in test_indices]
    test_activities = [int(activities[i]) for i in test_indices]
    test_activities = one_hot_vector(test_activities, m="a")
    test_performers = [int(performers[i]) for i in test_indices]
    test_performers = one_hot_vector(test_performers, m="p")
    test_data_aligned = aligned_data[test_indices]

    # dump the test sequences in a txt file
    test_samples = [sequences[i] for i in test_indices]
    with open(osp.join(base_folder, "statistics", "test_samples.txt"), "w") as f:
        for sample in test_samples:
            f.write(f"{sample}\n")

    train_data, val_data, folds = split_train_val_data(aligned_data, train_val_samples, train_val_activities, train_val_performers, n_folds=n_folds)

    test_data = {}
    test_data["data"] = test_data_aligned
    test_data["activities"] = test_activities
    test_data["performers"] = test_performers

    return train_data, val_data, test_data, folds

def save_dataset(test_data, train_data, val_data, folds, n_folds=4):
    with h5py.File("/u/home/aoma/gr/SGN/data/sitc/SITC_TC2-1.h5", 'w') as f:
        f.create_group("test")

        test_group = f["test"]
        test_group.create_dataset("x", data=test_data["data"])
        print(f"Test data shape: {test_data['data'].shape}")
        test_group.create_dataset("a", data=test_data["activities"])
        test_group.create_dataset("p", data=test_data["performers"])

        # add the train and val datasets for each fold
        for fold in range(n_folds):
            fold_group = f.create_group(f"f{fold}")

            train_group = fold_group.create_group("train")
            val_group = fold_group.create_group("val")

            train_group.create_dataset("x", data=train_data[fold])
            print(f"Train data shape: {train_data[fold].shape}")
            train_a_labels = one_hot_vector(folds[fold]["train_activities"], m="a")
            train_group.create_dataset("a", data=train_a_labels)
            train_p_labels = one_hot_vector(folds[fold]["train_performers"], m="p")
            train_group.create_dataset("p", data=train_p_labels)

            val_group.create_dataset("x", data=val_data[fold])
            print(f"Val data shape: {val_data[fold].shape}")
            val_a_labels = one_hot_vector(folds[fold]["val_activities"], m="a")
            val_group.create_dataset("a", data=val_a_labels)
            val_p_labels = one_hot_vector(folds[fold]["val_performers"], m="p")
            val_group.create_dataset("p", data=val_p_labels)


if __name__ == '__main__':
    with open(osp.join(base_folder, "statistics", "sequences.txt"), "r") as f:
        sequences = f.readlines()

    activities_include = ["PREPARE_MEAL", "BRUSH_TEETH"] # activities to include
    performers_include = [i for i in range(1, 17) if i not in [1,3,5,6,14,15,16]]

    sequences = [x.strip() for x in sequences]
    performers = [x[1:4] for x in sequences]
    activities = [x[5:8] for x in sequences]

    updated_sequences = []
    updated_performers = []
    updated_activities = []
    for i, sequence in enumerate(sequences):
        activity = activities[i]
        activity_name = list(activity_mappings.keys())[list(activity_mappings.values()).index(int(activity))]
        performer_id = int(performers[i])
        if activity_name in activities_include and performer_id in performers_include:
            updated_sequences.append(sequence)
            updated_performers.append(performers[i])
            updated_activities.append(activities[i])

    activities = updated_activities
    performers = updated_performers
    sequences = updated_sequences

    print(f"Using activities: {np.unique(activities)}")
    activity_mapping = {activity: idx for idx, activity in enumerate(np.unique(activities))}
    activities = np.array([activity_mapping[x] for x in activities], dtype=np.int32)
    print(f"Using performers: {np.unique(performers)}")
    performer_mapping = {performer: idx for idx, performer in enumerate(np.unique(performers))}
    performers = np.array([performer_mapping[x] for x in performers], dtype=np.int32)

    reformatted_data = reformat_pkl_data(sequences, activity_mapping)
    aligned_data = align_frames(reformatted_data)

    train_data, val_data, test_data, folds = split_data(aligned_data, sequences, activities, performers, n_folds=4)
    #save_dataset(test_data, train_data, val_data, folds, n_folds=4)
    print("Done!")