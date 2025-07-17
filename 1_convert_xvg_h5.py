import os
import numpy as np
import h5py

# Set the directory to the location of the notebook
notebook_directory = os.getcwd()

# List all .xvg files in the current directory
trajectory_files = [f for f in os.listdir(notebook_directory) if f.endswith('.xvg')]

# Loop through all .xvg files
for i, selected_file in enumerate(trajectory_files, 1):
    try:
        print(f"Processing {i} : {selected_file}")

        # Import the selected .xvg file as a tab-separated values (TSV) file
        input_trajectory = np.loadtxt(selected_file)

        # Check the dimensions of the imported data
        Nframes, dimTraj = input_trajectory.shape
        Nparticles = (dimTraj - 1) // 3

        # Debugging: Print dimensions
        print(f"File: {selected_file}, Nframes: {Nframes}, dimTraj: {dimTraj}, Nparticles: {Nparticles}")

        # Check if the number of columns is as expected
        if (dimTraj - 1) % 3 != 0:
            raise ValueError(f"Unexpected number of columns in {selected_file}")

        # Extract time and trajectory data for X, Y, and Z coordinates
        time = input_trajectory[:, 0]
        
        trajectoryX = np.zeros((Nframes, Nparticles))
        trajectoryY = np.zeros((Nframes, Nparticles))
        trajectoryZ = np.zeros((Nframes, Nparticles))

        for j in range(Nparticles):
            trajectoryX[:, j] = input_trajectory[:, 1 + j * 3]
            trajectoryY[:, j] = input_trajectory[:, 2 + j * 3]
            trajectoryZ[:, j] = input_trajectory[:, 3 + j * 3]

        # Use the input file name (without extension) to construct the output HDF5 file name
        fn_hdf5 = os.path.join(notebook_directory, f"{os.path.splitext(selected_file)[0]}.h5")

        # Debugging: Print the HDF5 file path
        print(f"HDF5 File Path: {fn_hdf5}")

        # Export data to HDF5 format
        with h5py.File(fn_hdf5, 'w') as hdf:
            hdf.create_dataset('time', data=time)
            hdf.create_dataset('trajectoryX', data=trajectoryX)
            hdf.create_dataset('trajectoryY', data=trajectoryY)
            hdf.create_dataset('trajectoryZ', data=trajectoryZ)

        print(f"Data for {selected_file} successfully exported to {fn_hdf5}")

    except Exception as e:
        print(f"Error processing {selected_file}: {e}")

print("All files processed and exported to HDF5 format.")
