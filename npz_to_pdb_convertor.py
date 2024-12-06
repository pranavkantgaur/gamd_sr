import numpy as np
import os

# Set the input directory containing your .npz files
input_directory = '/home/pranav/gamd_sr/openmm_data_generation/lj_data_ours/run_5/'
# Set the output directory where you want to save .pdb files
output_directory = 'top_pymol_ours'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all .npz files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.npz'):
        # Extract simulation_id and frame_id from the filename
        parts = filename.split('_')
        if len(parts) != 3:
            continue  # Skip files that do not match the expected naming convention
        
        simulation_id = parts[1]  # Get the simulation ID
        frame_id = parts[2].replace('.npz', '')  # Get the frame ID without extension

        # Load the .npz file
        data = np.load(os.path.join(input_directory, filename))
        positions = data['pos']  # Assuming 'pos' is the key for positions

        # Create a subdirectory for each simulation if it doesn't exist
        sim_output_dir = os.path.join(output_directory, f'simulation_{simulation_id}')
        os.makedirs(sim_output_dir, exist_ok=True)

        # Save positions to a .pdb file in the corresponding subdirectory
        pdb_filename = f'frame_{frame_id}.pdb'
        with open(os.path.join(sim_output_dir, pdb_filename), 'w') as f:
            for i, pos in enumerate(positions):
                f.write(f"ATOM  {i+1:5d}  Ar   RES A   1    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}\n")
            f.write("END\n")

print("Conversion complete.")
