import re
import sys

def update_dataset_paths(config_file, mt_dataset_dir, mt_dataset, do_dataset_dir, do_dataset):
    # Read the config file content
    with open(config_file, 'r') as f:
        config_content = f.read()

    # Replace the paths for mt_dataset and do_dataset using regex
    updated_config_content = re.sub(
        r'("'+ mt_dataset + r'":\s*").*?(")', 
        r'\1' + mt_dataset_dir + r'\2', 
        config_content
    )

    updated_config_content = re.sub(
        r'("'+ do_dataset + r'":\s*").*?(")', 
        r'\1' + do_dataset_dir + r'\2', 
        updated_config_content
    )

    # Write the updated content back to the config file
    with open(config_file, 'w') as f:
        f.write(updated_config_content)

    print(f"Updated paths for {mt_dataset} and {do_dataset} in {config_file}")

if __name__ == "__main__":
    config_file = sys.argv[1]  # config file path
    mt_dataset_dir = sys.argv[2]  # MT dataset directory path
    mt_dataset = sys.argv[3]  # MT dataset name (e.g., 'food101')
    do_dataset_dir = sys.argv[4]  # DO dataset directory path
    do_dataset = sys.argv[5]  # DO dataset name (e.g., 'upmcfood101')

    update_dataset_paths(config_file, mt_dataset_dir, mt_dataset, do_dataset_dir, do_dataset)
