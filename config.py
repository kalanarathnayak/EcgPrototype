# Data paths and constants
DATA_PATH = 'input/mit-bih-arrhythmia-database/'

# Sampling and window parameters
FS = 360  # Sampling frequency
WINDOW_SIZE = 180  # Window size (180 samples)

# Model parameters
INPUT_SIZE = 180
D_MODEL = 256
NHEAD = 16
NUM_TRANSFORMER_LAYERS = 3
DROPOUT = 0.15
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005

# List of patients excluding records 102, 104, 107, and 217 (poor quality by AAMI standards)
PATIENTS = ['100', '101', '103', '105', '106', '108', '109',
            '111', '112', '113', '114', '115', '116', '117',
            '118', '119', '121', '122', '123', '124', '200',
            '201', '202', '203', '205', '207', '208', '209',
            '210', '212', '213', '214', '215', '219', '220',
            '221', '222', '223', '228', '230', '231', '232',
            '233', '234']