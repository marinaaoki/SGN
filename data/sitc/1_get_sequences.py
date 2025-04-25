import glob
import os.path as osp
import pickle

from mappings import activity_mappings, camera_mappings, data_dir, base_folder

def save_sequence(activity_pkl):
    file_name = osp.basename(activity_pkl)
    person_id = file_name[1:4]
    activity_name = file_name[5:-4]

    camera_id = str(camera_mappings[activity_name[:2]]).zfill(3)

    activity_name = activity_name[3:]
    activity_id = str(activity_mappings[activity_name]).zfill(3)

    with open(activity_pkl, "rb") as f:
        activity_data = pickle.load(f)

    for repl in activity_data.keys():
        if activity_data[repl] == {}:
            continue
        sequence_name = f"P{person_id}A{activity_id}C{camera_id}R{str(repl).zfill(3)}"
        with open(osp.join(base_folder, "statistics", "sequences.txt"), "a") as f:
            f.write(sequence_name + "\n")


if __name__ == "__main__":
    activities_per_person = glob.glob(osp.join(data_dir, "*.pkl"))
    activities_per_person = sorted(activities_per_person)

    for activity_pkl in activities_per_person:
        save_sequence(activity_pkl)

    print("Done!")