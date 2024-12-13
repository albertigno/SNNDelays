import os

# Source directory
source_dir = r'C:\Users\Alberto\abl1_shd50_rnn_8'

# Iterate through all files in the directory
for filename in os.listdir(source_dir):
    # Check if the filename contains '_rpt' 
    if '_rpt' in filename:
        # Create new filename with replacements
        new_filename = filename.replace('_rpt0_', '_rpt5_')
        new_filename = new_filename.replace('_rpt1_', '_rpt6_')
        new_filename = new_filename.replace('_rpt2_', '_rpt7_')
        new_filename = new_filename.replace('_rpt3_', '_rpt8_')
        new_filename = new_filename.replace('_rpt4_', '_rpt9_')
      
        # Construct full file paths
        old_path = os.path.join(source_dir, filename)
        new_path = os.path.join(source_dir, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')

print('Renaming complete!')