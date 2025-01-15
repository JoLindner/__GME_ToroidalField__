import numpy as np
import os
from GeneralMatrixElements_Parallel import frequencies_GYRE
from GeneralMatrixElements_Parallel import load_index_map_from_file
from config import ConfigHandler


def quasi_degenerate(l,n, eigentag=None):
    try:
        if eigentag is None or eigentag == 'Full':
            name_string = f'supermatrix_array_{l}_{n}_full.txt'
        elif eigentag == 'FirstApprox':
            name_string = f'supermatrix_array_{l}_{n}_first_approx.txt'
        elif eigentag == 'SelfCoupling':
            name_string = f'supermatrix_array_{l}_{n}_self_coupling.txt'
        else:
            # Raise an error if the eigen_tag is invalid
            raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')

        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', ConfigHandler().get('ModelConfig', 'model_name'), 'Supermatrices', name_string)
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"Supermatrix array file not found: {DATA_DIR}. Please check if the supermatrix array file was created.")

        supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')

        if supermatrix_array.size == 0:
            raise ValueError(f"Supermatrix array in {DATA_DIR} is empty or invalid. Please verify the file contents.")

        eigenvalues, eigenvectors = np.linalg.eig(supermatrix_array)
        return eigenvalues, eigenvectors

    except ValueError as ve:
        print(f"Value Error: {ve}")
        raise ve
    except FileNotFoundError as fnfe:
        print(f"File Not Found Error: {fnfe}")
        raise fnfe
    except Exception as e:
        print(f"An unexpected error occurred in calculating the eigenvectors and eigenvalues of the supermatrix: {e}")
        raise e


def frequency_shifts(eigenvalues,l,n):
    omega_ref = 2*np.pi*frequencies_GYRE(l,n)
    freq_shifts = eigenvalues/(2*omega_ref*2*np.pi)*10**3 #nHz
    return freq_shifts


def extract_indices(eigenvalues, eigenvectors,l,n, eigentag):
    try:
        # Load index_map
        if eigentag is None or eigentag == 'Full':
            name_string = f'index_map_supermatrix_array_{l}_{n}_full.txt'
        elif eigentag == 'FirstApprox':
            name_string = f'index_map_supermatrix_array_{l}_{n}_first_approx.txt'
        elif eigentag == 'SelfCoupling':
            name_string = f'index_map_supermatrix_array_{l}_{n}_self_coupling.txt'
        else:
            # Raise an error if the eigen_tag is invalid
            raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')

        data_dir = os.path.join(os.path.dirname(__file__), 'Output', ConfigHandler().get('ModelConfig', 'model_name'), 'Supermatrices', 'IndexMaps', name_string)
        index_map = load_index_map_from_file(data_dir)

        # Extract indices
        l_to_eigenvalue = []
        n_to_eigenvalue = []
        m_to_eigenvalue = []
        for index, value in enumerate(eigenvalues):
            max_abs_index = np.argmax(abs(eigenvectors[:, index]))
            #a_max = abs(eigenvectors[max_abs_index, index])
            #lprime = index_map[:, max_abs_index]['l']
            #nprime = index_map[:, max_abs_index]['n']
            #mprime = index_map[:, max_abs_index]['m']
            l_to_eigenvalue.append(index_map[0, max_abs_index]['l'])
            n_to_eigenvalue.append(index_map[0, max_abs_index]['n'])
            m_to_eigenvalue.append(index_map[0, max_abs_index]['m'])

            #print(f'Eigenvalue {index}: {value:.2f}, l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}, Eigenvector {max_abs_index}: {a_max:.2f}')

        return l_to_eigenvalue, n_to_eigenvalue, m_to_eigenvalue

    except ValueError as ve:
        print(f"Value Error: {ve}")
        raise ve
    except FileNotFoundError as fnfe:
        print(f"File Not Found Error: {fnfe}")
        raise fnfe
    except Exception as e:
        print(f"An unexpected error occurred in calculating the eigenvectors and eigenvalues of the supermatrix: {e}")
        raise e


def calculate_safe_extract_freq_shifts(l,n, eigentag=None):
    try:
        # ONLY WORKS IF SUPERMATRIX_ARRAY EXISTS (check is done in quasi_degenerate() function)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = quasi_degenerate(l, n, eigentag)
        print(f'Eigenvalues and eigenvectors calculated for l={l} n={n}')

        # Calculate frequency shifts
        f_shifts = frequency_shifts(eigenvalues, l, n)
        # Assign correct indices (l, n, m) to frequency shifts
        f_l, f_n, f_m = extract_indices(eigenvalues, eigenvectors, l, n, eigentag)
        freq_shift_indexed = np.column_stack(
            (np.transpose(f_shifts), np.transpose(f_l), np.transpose(f_n), np.transpose(f_m)))
        dtype = [('freq', np.float64), ('l', np.int64), ('n', np.int64), ('m', np.int64)]
        freq_shift_indexed = np.array([tuple(row) for row in freq_shift_indexed], dtype=dtype)
        freq_shift_indexed = np.sort(freq_shift_indexed, order=['l', 'n', 'm'])
        filtered_freq_shift_indexed = freq_shift_indexed[(freq_shift_indexed['l'] == l) & (freq_shift_indexed['n'] == n)]
        print(f'Calculated frequency shifts and assigned indices')

        # Write result file
        if eigentag is None or eigentag == 'Full':
            name_string = f'freq_shifts_{l}_{n}_full.txt'
        elif eigentag == 'FirstApprox':
            name_string = f'freq_shifts_{l}_{n}_first_approx.txt'
        elif eigentag == 'SelfCoupling':
            name_string = f'freq_shifts_{l}_{n}_self_coupling.txt'
        else:
            # Raise an error if the eigen_tag is invalid
            raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')

        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', ConfigHandler().get("ModelConfig", "model_name"), 'FrequencyShifts', name_string)
        header = '#freq [nHz], l, n, m\n'
        with open(DATA_DIR, 'w') as file:
            file.write(header)
            np.savetxt(file, filtered_freq_shift_indexed, fmt='%.15f %d %d %d')
        print(f'Saved frequency shifts to {DATA_DIR}')

        # Return frequency shifts of modes given by (l, n, m)
        return filtered_freq_shift_indexed

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error in frequency shift calculation and index assignment: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred in calculate_safe_extract_freq_shifts: {e}")
        raise e


def main():
    pass
    '''
    # ONLY WORKS IF SUPERMATRIX_ARRAY EXISTS (check is done in quasi_degenerate() function)
    config = ConfigHandler('config.ini')
    eigentag = config.get("Eigenspace", "eigenspace_tag")
    
    l=5
    n=6
    calculate_safe_extract_freq_shifts(l,n, eigentag)
    n=12
    calculate_safe_extract_freq_shifts(l,n, eigentag)
    n=18
    calculate_safe_extract_freq_shifts(l,n, eigentag)
    '''


if __name__== '__main__':
    main()