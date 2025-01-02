import numpy as np
import os
from GeneralMatrixElements_Parallel import frequencies_GYRE

def quasi_degenerate(l,n):
    name_string = f'supermatrix_array_{l}_{n}_second_approx.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)
    supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')
    eigenvalues, eigenvectors = np.linalg.eig(supermatrix_array)
    return eigenvalues, eigenvectors

def frequency_shifts(eigenvalues,l,n):
    omega_ref = 2*np.pi*frequencies_GYRE(l,n)
    freq_shifts = eigenvalues/(2*omega_ref*2*np.pi)*10**3 #nHz
    return freq_shifts
def extract_indices(eigenvalues, eigenvectors,l,n):
    # Load index_map
    name_string = f'index_map_supermatrix_array_{l}_{n}_second_approx.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)
    index_map = load_index_map_from_file(data_dir)

    # Extract indices
    l_to_eigenvalue = []
    n_to_eigenvalue = []
    m_to_eigenvalue = []
    for index, value in enumerate(eigenvalues):
        max_abs_index = np.argmax(abs(eigenvectors[:, index]))
        a_max = abs(eigenvectors[max_abs_index, index])
        #lprime = index_map[:, max_abs_index]['l']
        #nprime = index_map[:, max_abs_index]['n']
        #mprime = index_map[:, max_abs_index]['m']
        l_to_eigenvalue.append(index_map[0, max_abs_index]['l'])
        n_to_eigenvalue.append(index_map[0, max_abs_index]['n'])
        m_to_eigenvalue.append(index_map[0, max_abs_index]['m'])

        #print(f'Eigenvalue {index}: {value:.2f}, l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}, Eigenvector {max_abs_index}: {a_max:.2f}')

    return l_to_eigenvalue, n_to_eigenvalue, m_to_eigenvalue

def load_index_map_from_file(filename):
    # Read lines from the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Remove newline characters
    lines = [line.strip() for line in lines]

    # Initialize matrix size based on the number of lines
    matrix_size = len(lines)
    index_map = np.empty((matrix_size, matrix_size), dtype=np.dtype([('l', np.int64), ('n', np.int64), ('m', np.int64), ('lprime', np.int64), ('nprime', np.int64), ('mprime', np.int64)]))

    # Parse each line and populate index_map
    for i, line in enumerate(lines):
        # Split line by commas to get tuples as strings
        tuple_strs = line.split(', ')
        tuples = []

        # Parse each tuple string into a tuple of integers
        for tuple_str in tuple_strs:
            tuple_vals = tuple(map(int, tuple_str.split()))
            tuples.append(tuple_vals)

        # Assign parsed tuples to index_map
        index_map[i] = tuples

    return index_map

def save_index_map_to_file(index_map, l, n):
    #INDEX MAP FOR APPROXIMATION
    name_string = f'index_map_supermatrix_array_{l}_{n}_second_approx.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)

    with open(data_dir, 'w') as f:
        for row in index_map:
            row_str = ", ".join(f"{col['l']} {col['n']} {col['m']} {col['lprime']} {col['nprime']} {col['mprime']}" for col in row)
            f.write(row_str + "\n")

    print(f"Index map saved to {data_dir}")


def calculate_safe_extract_freq_shifts(l,n):
    #ONLY WORKS IF SUPERMATRIX_ARRAY_FIRST_APPROX EXISTS
    eigenvalues, eigenvectors = quasi_degenerate(l, n)
    print(f'Eigenvalues and eigenvectors calculated for l={l} n={n}')
    f_shifts = frequency_shifts(eigenvalues, l, n)
    f_l, f_n, f_m = extract_indices(eigenvalues, eigenvectors, l, n)
    freq_shift_indexed = np.column_stack(
        (np.transpose(f_shifts), np.transpose(f_l), np.transpose(f_n), np.transpose(f_m)))
    dtype = [('freq', np.float64), ('l', np.int64), ('n', np.int64), ('m', np.int64)]
    freq_shift_indexed = np.array([tuple(row) for row in freq_shift_indexed], dtype=dtype)
    freq_shift_indexed = np.sort(freq_shift_indexed, order=['l', 'n', 'm'])
    filtered_freq_shift_indexed = freq_shift_indexed[(freq_shift_indexed['l'] == l) & (freq_shift_indexed['n'] == n)]
    print(f'Calculated frequency shifts and assigned indices')
    name_string = f'freq_shifts_{l}_{n}_second_approx.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output', 'Frequency_shifts', name_string)
    header = '#freq [nHz], l, n, m\n'
    with open(data_dir, 'w') as file:
        file.write(header)
        np.savetxt(file, filtered_freq_shift_indexed, fmt='%.15f %d %d %d')
    print(f'Saved frequency shifts to {data_dir}')
    return filtered_freq_shift_indexed


def main():
    l=5
    n=6
    calculate_safe_extract_freq_shifts(l,n)
    n=12
    calculate_safe_extract_freq_shifts(l,n)
    n=18
    calculate_safe_extract_freq_shifts(l,n)

if __name__== '__main__':
    main()