import sys
import numpy as np

def parse_dh_param_file(dh_config_file):
    assert(dh_config_file is not None)
    f_line_contents = None
    with open(dh_config_file, "r") as f:
        f_line_contents = f.readlines()

    assert(f.closed)
    assert(f_line_contents is not None)
    # maybe not the most efficient/clean/etc. way to do this, but should only have to be done once so NBD
    dh_params = np.asarray([line.rstrip().split(',') for line in f_line_contents[1:]])
    dh_params = dh_params.astype(float)
    return dh_params


### TODO: parse a pox parameter file
def parse_pox_param_file(pox_config_file):
    """
    Parse a POX parameter file to extract the M matrix and screw vectors.

    Parameters:
        pox_config_file (str): Path to the POX configuration file.

    Returns:
        tuple: A tuple containing:
            - M (numpy.ndarray): The 4x4 transformation matrix.
            - S (numpy.ndarray): The 6xn screw vectors matrix.
    """
    with open(pox_config_file, 'r') as file:
        lines = file.readlines()
    
    m_matrix = []
    screw_vectors = []
    mode = None

    for line in lines:
        line = line.strip()
        if line.startswith("# M matrix"):
            mode = "M"
            continue
        elif line.startswith("# screw vectors"):
            mode = "S"
            continue
        elif line == "" or line.startswith("#"):  # Skip empty lines or comments
            continue
        elif mode == "M":
            m_matrix.append([float(x) for x in line.split()])
        elif mode == "S":
            screw_vectors.append([float(x) for x in line.split()])

    # Convert to numpy arrays
    M = np.array(m_matrix)
    S = np.array(screw_vectors)  # Transpose so screw vectors are columns

    return M, S