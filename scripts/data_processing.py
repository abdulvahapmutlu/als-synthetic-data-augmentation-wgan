# --------------------
# 1. Data Loading & Preprocessing
# --------------------

# Function to load and preprocess EEG data from .mat files
def load_eeg_data(class_0_path, class_1_path):
    eeg_data_list = []
    labels_list = []

    # Helper function to extract EEG data from a .mat file
    def extract_eeg_data(file_path):
        mat_data = scipy.io.loadmat(file_path)
        
        # Find the correct key for EEG data (excluding metadata keys)
        data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        if len(data_keys) == 0:
            raise ValueError(f"No valid EEG data found in {file_path}")

        # Assuming the first valid key contains the EEG data
        eeg_data = mat_data[data_keys[0]]
        
        # Check if it's a numeric matrix
        if not isinstance(eeg_data, np.ndarray):
            raise ValueError(f"Unexpected data type in {file_path}: {type(eeg_data)}")
        
        # Normalize the data between -1 and 1
        eeg_data = eeg_data.astype('float32')
        eeg_data = (eeg_data - np.min(eeg_data)) / (np.max(eeg_data) - np.min(eeg_data)) * 2 - 1
        return eeg_data

    # Load class 0 data (files starting with '0_')
    for file_name in os.listdir(class_0_path):
        if file_name.endswith('.mat') and file_name.startswith('0_'):
            file_path = os.path.join(class_0_path, file_name)
            eeg_data = extract_eeg_data(file_path)
            eeg_data_list.append(eeg_data)
            labels_list.extend([0] * eeg_data.shape[0])  # Append 0 label for each signal in the file

    # Load class 1 data (files starting with '1_')
    for file_name in os.listdir(class_1_path):
        if file_name.endswith('.mat') and file_name.startswith('1_'):
            file_path = os.path.join(class_1_path, file_name)
            eeg_data = extract_eeg_data(file_path)
            eeg_data_list.append(eeg_data)
            labels_list.extend([1] * eeg_data.shape[0])  # Append 1 label for each signal in the file

    # Stack all EEG data and labels into NumPy arrays
    eeg_data = np.vstack(eeg_data_list)
    labels = np.array(labels_list)
    return eeg_data, labels

# Define the paths to the folder with .mat files for each class
folder_0 = r'C:\Users\offic\OneDrive\Masaüstü\datasets\ALS\0'
folder_1 = r'C:\Users\offic\OneDrive\Masaüstü\datasets\ALS\1'

# Load EEG data and labels
eeg_data, labels = load_eeg_data(folder_0, folder_1)
eeg_signal_length = 32  # Each signal is of length 32

print(f"Loaded {eeg_data.shape[0]} samples with signal length {eeg_signal_length}")
print(f"Labels shape: {labels.shape}")
