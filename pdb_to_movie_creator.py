import os
from pymol import cmd

# Set the input directory containing your .pdb files
input_directory = '../top_pymol/simulation_0'
# Set the output directory where you want to save the movie
output_directory = '../movie_dir'
# Define the output movie filename
output_movie_filename = os.path.join(output_directory, 'simulation_movie.mp4')

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Clear existing objects in PyMOL
cmd.reinitialize()

# Load all .pdb files from the input directory
pdb_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.pdb')])[:50]


# Create scenes for each frame
for i, pdb_file in enumerate(pdb_files):
    
    # Clear any previously loaded frames
    cmd.delete('all')  # Remove all previous objects from the scene
    
    # Load the current frame
    cmd.load(os.path.join(input_directory, pdb_file), f'frame_{i}')
    
   
    # Optionally set the view or other properties here
    cmd.show('spheres', f'frame_{i}')  # Show as spheres (or other representation)
    
    # You can customize colors or representations if needed
    #cmd.color('blue', f'frame_{i}')  # Color all frames blue
    
    # Create a new scene for this frame
    cmd.scene(f'scene_{i}', 'store')  # 0 means current state    

    # Save the current scene as an image (PNG format)
    image_filename = os.path.join(output_directory, f'frame_{i}.png')
    cmd.png(image_filename, width=800, height=600, dpi=300)  # Adjust dimensions and DPI as needed

# Invoke FFmpeg to create a video from the saved frames
ffmpeg_command = f"ffmpeg -framerate 1 -i {output_directory}/frame_%d.png -c:v libx264 -pix_fmt yuv420p {output_movie_filename}"
os.system(ffmpeg_command)

print(f"Movie saved to {output_movie_filename}")
