import os
import h5py


# Parameters
TIME_STEPS = 50 # Number of time steps in each slice sample


# Data paths
# Source folder
SRC_FOLDER = 'data/large/superbench/superbench_v1/'
# Destination folder
DST_FOLDER = 'data/large/superbench/'


# Concate a list of datasets
def concat_datasets(src_paths, dst_path):
    src_data = []
    src_files = []
    # Open the source files
    for src_path in src_paths:
        src_file = h5py.File(src_path, 'r')
        src_files.append(src_file)
        # There will be only one dataset in the file
        for key in src_file.keys():
            src_data.append(src_file[key])
    # Open (or create) a destination file in write mode
    with h5py.File(dst_path, 'w') as dst_file:
        # Compute the resulting size after concatenation
        size = 0
        for dataset in src_data:
            size += dataset.shape[0]
        # Create a dataset in the destination file
        shape_ = src_data[0].shape
        dst_data = dst_file.create_dataset('data', 
            (size, shape_[1], shape_[2], shape_[3]), 
            dtype=src_data[0].dtype)
        # Concatenate
        start = 0
        for dataset in src_data:
            end = start + dataset.shape[0]
            dst_data[start:end] = dataset
            start = end
    # Close the source files
    for src_file in src_files:
        src_file.close()


# Compute the resulting size after slicing
def compute_slice_size(in_data):
    # Loop with window=TIME_STEPS and step=TIME_STEPS//2 
    # over the first dimension
    start = 0
    window = TIME_STEPS
    step = TIME_STEPS//2
    size = 0
    while start + window <= in_data.shape[0]:
        size += 1
        start += step
    if start < in_data.shape[0]:
        size += 1
    return size


# Slice with a stride of TIME_STEPS//2 and a window size of TIME_STEPS
# and permute the axis to (samples, height, width, time_steps, channels)
def slice_and_permute(src_path, dst_path):
    # Open the source file in read mode
    with h5py.File(src_path, 'r') as src_file:
        # There will be only one dataset in the file
        for key in src_file.keys():
            src_data = src_file[key]
        # Open (or create) a destination file in write mode
        with h5py.File(dst_path, 'w') as dst_file:
            # Compute the resulting size after slicing
            size = compute_slice_size(src_data)
            # Create a dataset in the destination file
            shape_ = src_data.shape
            dst_data = dst_file.create_dataset('data', 
                (size, shape_[2], shape_[3], TIME_STEPS, shape_[1]), 
                dtype=src_data.dtype)
            # Slice and permute
            start = 0
            window = TIME_STEPS
            step = TIME_STEPS//2
            while start + window <= src_data.shape[0]:
                slice_data = src_data[start:start+window]
                # Permute data
                permuted_data = slice_data.transpose((2, 3, 0, 1))
                # Write to the output dataset in the destination file
                dst_data[start//step] = permuted_data
                # Update start
                start += step
            if start < src_data.shape[0]:
                # Take the last elements of size window
                slice_data = src_data[-window:]
                # Permute data
                permuted_data = slice_data.transpose((2, 3, 0, 1))
                # Write to the output dataset in the destination file
                dst_data[-1] = permuted_data


# Prepocess a dataset
def preprocess(src_path, dst_path):
    print("======")
    print("Preprocessing", src_path, "to", dst_path)
    print("======")
    if isinstance(src_path, list):
        tmp_path = dst_path.replace('.hdf5', '_tmp.hdf5')
        concat_datasets(src_path, tmp_path)
        slice_and_permute(tmp_path, dst_path)
        os.remove(tmp_path)
    else:
        slice_and_permute(src_path, dst_path)


# Read .h5 file and get the shape
def get_shape(path):
    with h5py.File(path, 'r') as f:
        # There will be only one dataset in the file
        for key in f.keys():
            return f[key].shape


if __name__ == '__main__':
    # Walk through the SRC_FOLDER
    for root, dirs, files in os.walk(SRC_FOLDER):
        h5_files = [f for f in files if f.endswith('.h5')]

        # If there's more than one .h5 file in the folder of the same shape, 
        # set concat_flag to True
        concat_flag = False
        if len(h5_files) > 1:
            # Check if all the files have the same shape
            shapes = [get_shape(os.path.join(root, f)) for f in h5_files]
            if len(set(shapes)) == 1:
                concat_flag = True
                # Create a new file name
                # which is the concatenation of all the original file names
                # with '_' as the delimiter
                dst_path = os.path.join(root, '_'.join(h5_files))
                dst_path = dst_path.replace('.h5', '')
                dst_path += '.hdf5'

        if concat_flag:
            print('Concatenating files in', root)
            print('Destination path:', dst_path)
            print('File names:', h5_files)
            print('Shape:', shapes)
            
            # Preprocess the files
            src_paths = [os.path.join(root, f) for f in h5_files]

            # Construct the destination path by replacing SRC_FOLDER with DST_FOLDER
            dst_path = dst_path.replace(SRC_FOLDER, DST_FOLDER)
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            preprocess(src_paths, dst_path)
            continue

        for file in h5_files:
            src_path = os.path.join(root, file)
            
            # Construct the destination path by replacing SRC_FOLDER with DST_FOLDER
            dst_path = src_path.replace(SRC_FOLDER, DST_FOLDER)

            # Replace .h5 extension with .hdf5 in the destination path
            dst_path = dst_path.replace('.h5', '.hdf5')
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Preprocess the file
            preprocess(src_path, dst_path)
