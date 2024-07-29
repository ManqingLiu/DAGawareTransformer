import json
import os
from argparse import ArgumentParser

def modify_json_files(data_name, range_start, range_end):
    for i in range(range_start, range_end):
        file_name = f'{data_name}_nomask_sample{i}.json'
        with open(file_name, 'r') as file:
            data = json.load(file)

        # Modify the relevant fields
        if data_name == 'acic':
            data['cfcv']['training']['dag_attention_mask'] = 'False'

        # Write the modified data back to the file
        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4)

# Function to replace 'psid' with 'cps' in the given JSON content
def replace_psid_with_cps(dest_dir, num_files=50):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for i in range(num_files):
        dest_file = os.path.join(dest_dir, f"lalonde_cps_sample{i}.json")

        with open(dest_file, 'r') as f:
            data = json.load(f)

        # Convert the JSON data to a string and replace "psid" with "cps"
        data_str = json.dumps(data)
        data_str = data_str.replace("psid", "cps")

        # Convert the string back to JSON format
        data = json.loads(data_str)

        with open(dest_file, 'w') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True,
                        help="Directory containing the JSON config files.")
    args = parser.parse_args()
    #replace_psid_with_cps(args.config_dir, 50)
    # cd to config_dir
    os.chdir(args.config_dir)
    modify_json_files('acic', 1, 11)
