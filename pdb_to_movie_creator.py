import os
from pymol import cmd

# Set the input directory containing your .pdb files
input_directory = '../top_pymol_ours/simulation_0'
# Set the output directory where you want to save the movie
output_directory = '.'
# Define the output movie filename
output_movie_filename = os.path.join(output_directory, 'simulation_movie.mp4')

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Clear existing objects in PyMOL
cmd.reinitialize()

# Load all .pdb files from the input directory
pdb_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.pdb')])

# Create scenes for each frame
for i, pdb_file in enumerate(pdb_files):
    # Load the current frame
    cmd.load(os.path.join(input_directory, pdb_file), f'frame_{i}')
    
    # Create a new scene for this frame
    cmd.scene(f'scene_{i}', 'store')  # 0 means current state
    
    # Optionally set the view or other properties here
    cmd.show('spheres', f'frame_{i}')  # Show as spheres (or other representation)
    
    # You can customize colors or representations if needed
    cmd.color('blue', f'frame_{i}')  # Color all frames blue

# Set up the timeline with a 1-second offset between scenes
for i in range(len(pdb_files)):
    cmd.scene(f'scene_{i}', i)  # Each scene corresponds to its index

# Save the movie with 1 second per frame
cmd.mplay()  # Start playing the movie in PyMOL

# Save the movie to an output file
cmd.save(output_movie_filename)

print(f"Movie saved to {output_movie_filename}")
