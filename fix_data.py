import os
import json

data_folder = "./data/BIWI_prepared"
files = os.listdir(data_folder)
files = [os.path.join(data_folder, f) for f in files if f.lower().endswith(".json")]


for file in files:
    with open(file) as json_file:
        data = json.load(json_file)
        data["yaw"] = data["yaw"][0]
        data["pitch"] = data["pitch"][0]
        data["roll"] = data["roll"][0]

    with open(file, 'w') as outfile:
        json.dump(data, outfile)