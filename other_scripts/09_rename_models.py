import os

# Source directory
source_dir = r'C:\Users\Alberto\checkpoints\abl1_shd50_rnn'

# Iterate through all files in the directory
for filename in os.listdir(source_dir):
    # Check if the filename contains '_rpt' 
    if '_rpt' in filename:
        # Create new filename with replacements
        new_filename = filename.replace('_rpt0_', '_rpt8_').replace('_rpt1_', '_rpt9_')
        
        # Construct full file paths
        old_path = os.path.join(source_dir, filename)
        new_path = os.path.join(source_dir, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')

print('Renaming complete!')