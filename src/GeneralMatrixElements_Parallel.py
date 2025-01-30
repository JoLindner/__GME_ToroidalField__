import numpy as np
import angular_kernels
import radial_kernels
import os
import pygyre as pg
import concurrent.futures
import time
from config import ConfigHandler
import h5py
import glob
import itertools
from scipy import optimize
import scipy.interpolate


def single_GME(l, n, m, lprime, nprime, mprime, magnetic_field_s, magnetic_field_sprime=None, model_name=None, mesa_data=None, temp_name=None):
    # Output directory initialization
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', model_name, 'GeneralMatrixElements')

    # Path initialization
    main_db_path = os.path.join(DATA_DIR, 'main_gme_results.h5')
    temp_file_name = f'temp_gme_results_{temp_name}.h5'
    temp_db_path = os.path.join(DATA_DIR, 'Temp', temp_file_name)
    try:
        try:
            # Check if the result already exists in the main dataset
            with h5py.File(main_db_path, 'r') as main_hdf:
                result_key = f'/gme/{l}_{n}_{m}_{lprime}_{nprime}_{mprime}'
                if result_key in main_hdf:
                    result = main_hdf[result_key][()]
                    if result is not None:
                        print(f'GME already exists: l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}, result={result}')
                        # Return the existing result if found in the main DB
                        return result
        except OSError:
            # Main database file doesn't exist yet, just continue without an error
            pass

        # Magnetic fields
        if magnetic_field_sprime is None:
            magnetic_field_sprime = magnetic_field_s
        s = magnetic_field_s.s
        sprime = magnetic_field_sprime.s

        # First selection rule
        if m != mprime:
            return 0.0

        # Second selection rule
        if (l+lprime+s+sprime) % 2 == 1:
            return 0.0

        # Load mesa_data
        radius_array = mesa_data.radius_array
        deriv_lnRho = mesa_data.deriv_lnRho

        # Compute radial kernels
        radial_kernel_obj = radial_kernels.RadialKernels(l, n, lprime, nprime, radius_array, magnetic_field_s, magnetic_field_sprime, deriv_lnRho)

        # Compute angular kernels
        angular_kernels_obj = angular_kernels.AngularKernels(lprime, l, s, sprime, mprime, m)

        try:
            # H_k',k; GME computation:
            general_matrix_element=1/(4*np.pi)*(radial_kernel_obj.R1*angular_kernels_obj.s1
                                            +radial_kernel_obj.R2*(angular_kernels_obj.s2-angular_kernels_obj.s5)
                                            -radial_kernel_obj.R3*(angular_kernels_obj.s3+angular_kernels_obj.s6)
                                            +radial_kernel_obj.R4*angular_kernels_obj.s4
                                            +radial_kernel_obj.R5*(angular_kernels_obj.s7+angular_kernels_obj.s8)
                                            +radial_kernel_obj.R6*(angular_kernels_obj.s9-angular_kernels_obj.s10+angular_kernels_obj.s11
                                            -2*angular_kernels_obj.s13+angular_kernels_obj.s14-angular_kernels_obj.s15-angular_kernels_obj.s16
                                            -angular_kernels_obj.s18-angular_kernels_obj.s19-angular_kernels_obj.s20-angular_kernels_obj.s22
                                            -angular_kernels_obj.s23)
                                            -radial_kernel_obj.R7*angular_kernels_obj.s17
                                            -radial_kernel_obj.R8*angular_kernels_obj.s21)

            if general_matrix_element is None:
                raise ValueError(f"The computed GME is None for the input parameters: l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}.")

        except Exception as e:
            print(f"Failed to compute GME for the input parameters l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}. Error: {e}")
            raise e

        try:
            # Create and write to temporary file
            with h5py.File(temp_db_path, 'a') as temp_hdf:
                # Saving the computed GME to the temporary file
                if 'gme' not in temp_hdf:
                    temp_hdf.create_group('gme')
                dataset_path = f'/gme/{l}_{n}_{m}_{lprime}_{nprime}_{mprime}'
                temp_hdf.create_dataset(dataset_path, data=general_matrix_element)
                print(f'Saved GME: l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}, result={general_matrix_element}')

        except Exception as e:
            print(f"Error during file operations: {e}")
            raise e

        return general_matrix_element

    except Exception as e:
        # Catch errors (e.g., programming errors, valueError)
        print(f"An error occurred in the single_GME function: {e}")
        raise e  # Re-raise to propagate the error


def normalization(l,n,mesa_data):
    #Get MESA structural data
    radius_array = mesa_data.radius_array
    r_sun = mesa_data.R_sun  # r_sun in cm
    rho_0 = mesa_data.rho_0*r_sun**3 #g/R_sun^3

    #Calculate function to integrate
    func = rho_0*(radial_kernels.eigenfunctions(l,n,radius_array)[0]**2+l*(l+1)*radial_kernels.eigenfunctions(l,n,radius_array)[1]**2)*radius_array**2
    # Perform the radial integration
    radial_integration_result = radial_kernels.radial_integration(radius_array, func) #g*R_sun^2

    if radial_integration_result is None:
        raise ValueError(f"Radial integration for normalization function failed for l={l} and n={n} and resulted in None.")

    return radial_integration_result  # g*R_sun^2


def frequencies_GYRE(l,n):
    try:
        # Initialize configuration handler
        config = ConfigHandler()

        #Read path to summary file
        summary_GYRE_path = config.get("StellarModel", "summary_GYRE_path")
        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', summary_GYRE_path)
        summary_file = pg.read_output(DATA_DIR)
        l_group = summary_file.group_by('l')
        filterd_freq=next(value for value in l_group.groups[l] if value['n_pg']==n)['freq'].real

    except FileNotFoundError as e:
        error_message = f"File not found: {DATA_DIR}. Please check if the summary file exists in the 'Data/GYRE' directory."
        raise FileNotFoundError(error_message)
    except KeyError as e:
        error_message = f"Key error: The expected data for l={l} and n={n} could not be found. Missing key: {e.args[0]}"
        raise KeyError(error_message)
    except StopIteration:
        #error_message = f"No data found for l={l} and n={n}. Please check the input values."
        #raise
        return None  # Return None, e.g. if l and n are out of range of existing frequencies
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        raise Exception(error_message)

    return filterd_freq  #microHz


def eigenspace(l, n, delta_freq_quadrat, eigentag=None, first_approx_data=None):
    try:
        # TAGS: Default/Full: full eigenspace; FirstApprox: first approximation; SelfCoupling: Only self-coupling (second Approximation)

        # doesn't matter here if I calculate the eigenspaces from the frequencies instead of angular frequencies
        freq_ref = frequencies_GYRE(l, n)

        # Initialize configuration handler
        config = ConfigHandler()

        # Read path to summary file
        summary_GYRE_path = config.get("StellarModel", "summary_GYRE_path")
        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', summary_GYRE_path)
        summary_file = pg.read_output(DATA_DIR)
        K_space = []

        if eigentag is None or eigentag == 'Full':
            #FULL EIGENSPACE
            #delta_freq in microHz
            for row in summary_file:
                if abs(row['freq'].real**2-freq_ref**2) <= delta_freq_quadrat and row['l'] > 1:   #exclude l>1, since eq. A.19 breaks down for these cases
                    new_row = {
                        'freq': row['freq'],
                        'n': row['n_pg'],
                        'l': row['l']}
                    K_space.append(new_row)

        elif eigentag == 'FirstApprox' and first_approx_data is not None:
            # Apply first approximation
            for (l, n), multiplet in first_approx_data['multiplets'].items():
                if abs(multiplet['freq'] ** 2 - freq_ref ** 2) <= delta_freq_quadrat:
                    new_row = {
                        'freq': multiplet['freq'],
                        'n': n,
                        'l': l
                    }
                    K_space.append(new_row)

        elif eigentag == 'SelfCoupling':
            # Only self-coupling (second approximation)
            for row in summary_file:
                if abs(row['freq'].real ** 2 - freq_ref ** 2) <= delta_freq_quadrat:
                    if l == row['l'] and n == row['n_pg']:
                        new_row = {
                            'freq': row['freq'],
                            'n': row['n_pg'],
                            'l': row['l']}
                        K_space.append(new_row)
        else:
            # Raise an error if the eigen_tag is invalid
            raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')

        return K_space

    except ValueError as ve:
        print(f"Error occurred in eigenspace creation: {ve}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred in the creation of the eigenspace: {e}")
        raise


def eigenspace_mode_search(l, n, freq_interval, first_approx_data=None):
    # Search quasi-degenerate multiplets with frequencies closer than 0.1 microHz to the reference multiplet
    # Doesn't matter here if I calculate the eigenspaces from the frequencies insteat of angular frequencies
    freq_ref = frequencies_GYRE(l, n)

    # Initialize configuration handler
    config = ConfigHandler()

    # Create eigenspace
    K_space = []
    for (l, n), multiplet in first_approx_data['multiplets'].items():
        if abs(multiplet['freq'] - freq_ref) <= freq_interval:
            new_row = {
                'freq': multiplet['freq'],
                'n': n,
                'l': l}
            K_space.append(new_row)

    return K_space


def supermatrix_element(omega_ref, l, n, m, lprime, nprime, mprime, magnetic_field_s, magnetic_field_sprime=None, model_name=None, mesa_data=None, temp_name=None):
    try:
        if magnetic_field_sprime is None:
            magnetic_field_sprime = magnetic_field_s

        # Fetch GME and Normalization factor
        gme = single_GME(l,n,m,lprime,nprime,mprime,magnetic_field_s,magnetic_field_sprime,model_name,mesa_data,temp_name)
        normal = normalization(l,n,mesa_data)

        # delta function
        delta = 1 if l == lprime and n == nprime and m == mprime else 0

        # includes conversion factor 10**12*10**6*R_sun to yield microHz^2
        sme = gme/normal*10**12*10**6*mesa_data.R_sun-(omega_ref**2-(2*np.pi*frequencies_GYRE(l,n))**2)*delta  # microHz^2

        if sme is None:
            raise ValueError(f"Supermatrix element (SME) is None for inputs: "
                             f"l={l}, n={n}, m={m}, l'={lprime}, n'={nprime}, m'={mprime}.")

        return sme, gme, normal

    except Exception as e:
        print(f"An error occurred in computing the 'supermatrix_element': {e}")
        raise e


def supermatrix_parallel_one_row(row, row_number, l, n, magnetic_field_s, magnetic_field_sprime=None, model_name=None, mesa_data=None, eigen_space=None):
    try:
        if magnetic_field_sprime is None:
            magnetic_field_sprime = magnetic_field_s
        kprime = row[0]
        mprime = row[1]
        omega_ref = 2*np.pi*frequencies_GYRE(l,n)
        K_space = eigen_space

        # Create for each thread unique temp name
        temp_name =f'sme_{l}_{n}_row_{row_number}'

        size = 0
        for i in range(0,len(K_space)):
            for iprime in range(0, len(K_space)):
                size = size+(2*K_space[i]['l']+1)*(2*K_space[iprime]['l']+1)
        if np.sqrt(size) % 1 == 0:
            matrix_size = int(np.sqrt(size))
        else:
            raise ValueError('Matrixsize not an integer')

        supermatrix_array_row = np.empty(matrix_size, dtype=np.float64)

        # fill supermatrix
        lprime = int(K_space[kprime]['l'])
        nprime = int(K_space[kprime]['n'])
        mprime_index = mprime
        mprime = mprime_index-lprime
        col = 0
        for k in range(0, len(K_space)):
            l = int(K_space[k]['l'])
            n = int(K_space[k]['n'])
            for m in range(0,2*l+1):
                m_index = m
                m = m_index-l
                #Z_k',k
                sme = supermatrix_element(omega_ref,l,n,m,lprime,nprime,mprime,magnetic_field_s, magnetic_field_sprime, model_name, mesa_data, temp_name)[0]
                supermatrix_array_row[col] = sme
                col += 1
        return np.transpose(supermatrix_array_row)

    except Exception as e:
        print(f"An error occurred in computing a row of the supermatrix: {e}")
        raise e


def supermatrix_parallel(l, n, magnetic_field_s, magnetic_field_sprime=None, eigen_space=None, mesa_data=None):
    # Initialize configuration handler
    config = ConfigHandler()
    # Load Model name
    model_name = config.get("ModelConfig", "model_name")
    # Check if Model name exists
    if not model_name:
        raise ValueError("The 'model_name' could not be fetched from the config.ini file under [ModelConfig] section.")

    try:
        if magnetic_field_sprime is None:
            magnetic_field_sprime = magnetic_field_s



        # Load eigenspace
        K_space = eigen_space

        # Create row indices
        rows = []
        for kprime in range(0, len(K_space)):
            l_aux = K_space[kprime]['l']
            for mprime in range(0, 2*l_aux+1):
                rows.append([kprime,mprime])

        # gets either SLURM_CPUS_PER_TASK or the number of CPU cores from the system
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

        # Add row_number to each row
        rows_with_numbers = [(index, row) for index, row in enumerate(rows)]

        # Execute parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                supermatrix_parallel_one_row,
                [row[1] for row in rows_with_numbers],  # The row itself (excluding the row_number)
                [row[0] for row in rows_with_numbers],  # row_number
                itertools.repeat(l),
                itertools.repeat(n),
                itertools.repeat(magnetic_field_s),
                itertools.repeat(magnetic_field_sprime),
                itertools.repeat(model_name),
                itertools.repeat(mesa_data),
                itertools.repeat(K_space)
            ))
        # Merge hdf5 files
        hdf5_merger(l, n, model_name)

        # Combine rows
        combined_result = np.vstack(results)
        return combined_result

    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        raise e


def hdf5_merger(l, n, model_name, max_retries=20, retry_interval=10):
    # Find files matching the pattern sme_{l}_{n}_row_*.h5
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', model_name, 'GeneralMatrixElements')
    main_db_path = os.path.join(DATA_DIR, 'main_gme_results.h5')
    temp_db_path = os.path.join(DATA_DIR, 'Temp')

    pattern = os.path.join(temp_db_path, f"temp_gme_results_sme_{l}_{n}_row_*.h5")
    temp_files = glob.glob(pattern)

    if not temp_files:
        print(f"No temporary HDF5 files matching pattern {pattern} were found.")
        return None

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Check if the main HDF5 file exists, create it if not
            if not os.path.exists(main_db_path):
                with h5py.File(main_db_path, 'w') as main_hdf:
                    # Create the top-level group
                    main_hdf.require_group('gme')
                print(f"Created main HDF5 file: {main_db_path}")

            # Open the main HDF5 file for merging
            with h5py.File(main_db_path, 'a') as main_hdf:
                main_group = main_hdf.require_group('gme')

                # Iterate over matching temp files
                for temp_file in temp_files:
                    with h5py.File(temp_file, 'r') as temp_hdf:
                        if 'gme' not in temp_hdf:
                            raise ValueError(f"No 'gme' group found in {temp_file}.")

                        temp_gme_group = temp_hdf['gme']
                        for dataset_name in temp_gme_group:
                            dataset_path = f'gme/{dataset_name}'

                            # If the dataset doesn't exist in the main file, copy it
                            if dataset_path not in main_hdf:
                                data = temp_gme_group[dataset_name][...]
                                main_group.create_dataset(dataset_name, data=data)
                                # print(f"Copied dataset {dataset_path} to main HDF5.")

            # If successful, remove temp files
            for temp_file in temp_files:
                os.remove(temp_file)
                # print(f"Deleted temporary file: {temp_file}")

            # Exit the loop after successful processing
            break

        except (OSError, BlockingIOError) as e:
            # Handle file locking or access issues
            retry_count += 1
            if retry_count < max_retries:
                print(f"Main HDF5 file locked. Retrying in {retry_interval} seconds... ({retry_count}/{max_retries})")
                time.sleep(retry_interval)
            else:
                print(f"Max retries reached. Could not access {main_db_path}.")
                raise e


def hdf5_investigate(file_path):
    try:
        with h5py.File(file_path, 'r') as hdf:
            print(f"\nInspecting HDF5 file: {file_path}")
            print("=" * 50)
            def print_hierarchy(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
                    # Read and preview a few data points
                    data_preview = obj[...]
                    if data_preview.size > 10:  # If too large, show first 10 elements
                        data_preview = data_preview.flat[:10]
                    print(f"{indent}  Preview: {data_preview}")

            hdf.visititems(print_hierarchy)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")


# INDEX MAP
def create_index_map(l,n, K_space):
    try:
        print(f'Create index map for l={l}, n={n}')

        # Compute matrix size
        size=0
        for i in range(0,len(K_space)):
            for iprime in range(0, len(K_space)):
                size=size+(2*K_space[i]['l']+1)*(2*K_space[iprime]['l']+1)

        # Check for square matrix dimension
        if np.sqrt(size) %1 ==0:
            matrix_size=int(np.sqrt(size))
        else:
            raise ValueError(f'Matrix size {size} calculated is not a square.')

        # Define structured array for index map
        dtype = np.dtype([('l', np.int64), ('n', np.int64), ('m', np.int64), ('lprime', np.int64), ('nprime', np.int64), ('mprime', np.int64)])
        index_map = np.empty((matrix_size,matrix_size), dtype=dtype)

        # Populate the index map
        row = 0
        for kprime in range(0, len(K_space)):
            lprime = K_space[kprime]['l']
            nprime = K_space[kprime]['n']
            for mprime in range(0, 2*lprime+1):
                mprime_index = mprime
                mprime = mprime_index-lprime
                col = 0
                for k in range(0, len(K_space)):
                    l = K_space[k]['l']
                    n = K_space[k]['n']
                    for m in range(0,2*l+1):
                        m_index = m
                        m = m_index-l
                        index_map[row,col] = (l,n,m,lprime,nprime,mprime)
                        col += 1
                row += 1

        print('Successfully created index map')
        return index_map

    except Exception as e:
        print(f"An unexpected error occurred in the creation of the index map: {e}")
        raise e


def save_index_map_to_file(index_map, l, n, eigentag=None):
    try:
        # Initialize configuration handler
        config = ConfigHandler()
        # Load Model name
        model_name = config.get("ModelConfig", "model_name")
        if not model_name:
            raise ValueError("The 'model_name' could not be fetched from the config.ini file under [ModelConfig] section.")


        if eigentag is None or eigentag == 'Full':
            name_string = f'index_map_supermatrix_array_{l}_{n}_full.txt'
        elif eigentag == 'FirstApprox':
            name_string = f'index_map_supermatrix_array_{l}_{n}_first_approx.txt'
        elif eigentag == 'SelfCoupling':
            name_string = f'index_map_supermatrix_array_{l}_{n}_self_coupling.txt'
        else:
            # Raise an error if the eigen_tag is invalid
            raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')

        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', model_name, 'Supermatrices', 'IndexMaps', name_string)

        with open(DATA_DIR, 'w') as f:
            for row in index_map:
                row_str = ", ".join(f"{col['l']} {col['n']} {col['m']} {col['lprime']} {col['nprime']} {col['mprime']}" for col in row)
                f.write(row_str + "\n")

        print(f"Index map saved to {DATA_DIR}")

    except Exception as e:
        print(f"An unexpected error occurred in saving the index map: {e}")
        raise e


def load_index_map_from_file(filename):
    try:
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

    except Exception as e:
        print(f"An unexpected error occurred in loading the index map: {e}")
        raise e


def first_approx(ltp, magnetic_field_s):
    # Checks for all multiplets in GYRE summary file with l>=2 and n>=0
    # Returns filtered list of multiplets with l>=2 and n>=0 within 3 sigma effective magnetic field range and threshhold radius

    # Compute 3sigma range of magnetic field:
    r_thresh = magnetic_field_s.mu + 3 * magnetic_field_s.sigma
    #print('Upper bound of 3 sigma effective magnetic field range: ', r_thresh, r'R_sun')

    # Filter out multiplets which do not penetrate into effective magnetic field range and skip l=1 (l=0 is not in r_ltp)
    multiplets_first_approx = {
        'multiplets': {
            (l, n): {'r_ltp': data['r_ltp'], 'freq': data['frequency']}
            for (l, n), data in ltp.items()
            if l != 1 and data['r_ltp'] <= r_thresh
        },
        'r_thresh': r_thresh
    }

    return multiplets_first_approx


def lower_turning_points(path_to_summary_file, mesa_data):
    # Read in summary file
    summary_file = pg.read_output(path_to_summary_file)
    l_group = summary_file.group_by('l')

    # Cubic Spline of sound speed
    c_sound_spline = scipy.interpolate.CubicSpline(mesa_data.radius_array, mesa_data.c_sound,
                                                   bc_type='natural')
    c_sound_spline_func = lambda x: c_sound_spline(x)

    # R_sun
    R_sun = mesa_data.R_sun

    # Lower turning point computation
    lower_tp = {}
    i = 0
    # print(optimize.root_scalar(lower_turning_point_fixed_point_func,method='secant', x0=0.5, bracket=[0,1], args=(l,n)))
    for l in l_group['l']:
        if l != 0:  # omit radial oscillations
            n_pg = l_group[i]['n_pg']
            if n_pg != 0:  # omit f modes
                r_ltp = optimize.root_scalar(lower_turning_point_fixed_point_func, method='toms748', x0=0.5,
                                             bracket=[0.001, 1], args=(i, l, l_group, R_sun, c_sound_spline_func))
                frequency = l_group[i]['freq']  # in microHz
                lower_tp[(int(l), int(n_pg))] = {
                    'r_ltp': r_ltp.root,
                    'frequency': frequency.real
                }
        i += 1

    return lower_tp


def lower_turning_point_fixed_point_func(x, index, l, l_group, R_sun, c_sound_spline_func):
    omega = 2*np.pi*l_group[index]['freq'].real*10**(-6)   # in Hz
    return omega**2/(l*(l+1))-(c_sound_spline_func(x))**2/(x*R_sun)**2


def main():
    # Initialize configuration handler
    config = ConfigHandler("config.ini")

    # Initialize data directory of GME
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', config.get("ModelConfig", "model_name"), 'GeneralMatrixElements')

    # Initialize stellar model (MESA data)
    mesa_data = radial_kernels.MesaData(config=config)

    # Initialize magnetic field parameters:
    B_max = config.getfloat("MagneticFieldModel", "B_max")
    mu = config.getfloat("MagneticFieldModel", "mu")
    sigma = config.getfloat("MagneticFieldModel", "sigma")
    s = config.getint("MagneticFieldModel", "s")
    sprime = config.getint("MagneticFieldModel", "sprime", default=s)

    # Initialize the magnetic field model
    magnetic_field_s = radial_kernels.MagneticField(B_max=B_max, mu=mu, sigma=sigma, s=s, radius_array=mesa_data.radius_array)
    magnetic_field_sprime = radial_kernels.MagneticField(B_max=B_max, mu=mu, sigma=sigma, s=sprime, radius_array=mesa_data.radius_array)

    ######################################
    # Compare Eigenspaces
    delta_freq_quadrat = config.getfloat("Eigenspace", "delta_freq_quadrat")  #microHz^2
    compare_eigenspace = False

    # Create first approximation data
    summary_dir = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', config.get("StellarModel", "summary_GYRE_path"))
    r_t = lower_turning_points(summary_dir, mesa_data)
    first_app = first_approx(r_t, magnetic_field_s)

    if compare_eigenspace:
        # reference multiplet:
        l, n = 9, 13
        print(f'Reference Frequency of multiplet (l={l}, n={n}): ', frequencies_GYRE(l, n), ' microHz')
        print(f'Eigenspace width: ', delta_freq_quadrat, ' microHz^2')
        # Eigenspace
        K_space_full = eigenspace(l, n, delta_freq_quadrat, eigentag='Full')
        print('Full: ', len(K_space_full), K_space_full)

        K_space_1approx = eigenspace(l, n, delta_freq_quadrat, eigentag='FirstApprox', first_approx_data=first_app)
        print('First Approx: ', len(K_space_1approx), K_space_1approx)

        K_space_selfcoupling = eigenspace(l, n, delta_freq_quadrat, 'SelfCoupling')
        print('Only Self-Coupling: ', len(K_space_selfcoupling), K_space_selfcoupling)

    ######################################
    # Test General Matrix Elements
    test_gme = False
    if test_gme:
        # Delete old temp test files:
        temp_path = os.path.join(DATA_DIR, 'Temp', 'temp_gme_results_test.h5')
        if os.path.isfile(temp_path):
            os.remove(temp_path)

        start_time = time.time()

        # Compute general matrix element::
        l, n, m = 2, 3, 1
        lprime, nprime, mprime = 2, 3, 1
        gme = single_GME(l,n,m,lprime,nprime,mprime,magnetic_field_s, model_name=config.get('ModelConfig', 'model_name'), mesa_data=mesa_data, temp_name='test')
        print(f'GME for l={l}, n={n}, m={m}, l\'={lprime}, n\'={nprime}, m\'={mprime}: ', gme, f' kg^2*R_sun^3')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed time in seconds: ', elapsed_time)

    ######################################
    # Investigate hdf5 files:
    investigate_hdf = False
    if investigate_hdf:
        main_db_path = os.path.join(DATA_DIR, 'main_gme_results.h5')
        # temp_db_path = os.path.join(DATA_DIR, 'Temp')

        hdf5_investigate(os.path.join(main_db_path))
        # hdf5_investigate(os.path.join(temp_db_path, 'temp_gme_results_sme_5_2_row_0.h5'))

        # Merge hdf5 files of same reference multiplet (l, n):
        # hdf5_merger(l=5, n=2, config.get("ModelConfig", "model_name"), max_retries=20, retry_interval=10)

    ######################################
    # Search for modes within a given frequency interval
    search_eigenspace = False
    if search_eigenspace:
        freq_interval = 0.005  # microHz

        # searches for eigenspaces containing 2 or more modes in the first approximation
        # eigenspace width is defined either by a frequency interval or delta_freq_quadrat
        n_len2, n_len2_quadrat = 0,0
        n_len3, n_len3_quadrat = 0,0
        n_len_larger, n_len_larger_quadrat = 0,0
        for l in range(2, 20):  # (0,150)
            for n in range(0, 36):  # (0,36)
                try:
                    K_space = eigenspace_mode_search(l,n, freq_interval, first_app)
                    K_space_quadrat = eigenspace(l, n, delta_freq_quadrat, eigentag='FirstApprox', first_approx_data=first_app)
                    if len(K_space) > 1:
                        if len(K_space) == 2:
                            n_len2 += 1
                            #print('K',K_space)
                        elif len(K_space) == 3:
                            n_len3 += 1
                            #print('K',K_space)
                        else:
                            n_len_larger += 1
                            #print('K',K_space)

                    if len(K_space_quadrat) > 1:
                        if len(K_space_quadrat) == 2:
                            n_len2_quadrat += 1
                            #print('K_qua',K_space_quadrat)
                        elif len(K_space_quadrat) == 3:
                            n_len3_quadrat += 1
                            print('K_qua',K_space_quadrat)
                        else:
                            n_len_larger_quadrat += 1
                            print('K_qua',K_space_quadrat)

                except:
                    continue
        print('Eigenspace with length 2: ',n_len2)
        print('Eigenspace with length 3: ',n_len3)
        print('Eigenspace with length larger 3: ',n_len_larger)
        print('Eigenspace quadrat with length 2: ',n_len2_quadrat)
        print('Eigenspace quadrat with length 3: ',n_len3_quadrat)
        print('Eigenspace quadrat with length larger 3: ',n_len_larger_quadrat)

    ######################################
    # Filter lower turning point
    lower_tp = False
    if lower_tp:
        l = 66
        n = 1
        filtered_r_ltp = r_t[(l, n)]['r_ltp']
        print(f"Lower turning point for (l,n) = ({l}, {n}):", filtered_r_ltp)

    ######################################
    # Test Kiefer and Roth 2018 approximation
    test_kiefer = False
    if test_kiefer:
        # Delete old temp test files:
        temp_path = os.path.join(DATA_DIR, 'Temp', 'temp_gme_results_test.h5')
        if os.path.isfile(temp_path):
            os.remove(temp_path)

        l, n, m = 5, 18, 0
        lprime, nprime, mprime = l, n, m
        angular_freq = 2 * np.pi * frequencies_GYRE(l, n)  # microHz

        # Quasi-degenerated perturbation theory:
        gme = single_GME(l, n, m, lprime, nprime, mprime, magnetic_field_s, magnetic_field_sprime, config.get('ModelConfig', 'model_name'), mesa_data=mesa_data, temp_name='test')
        normal = normalization(l, n, mesa_data)
        print('Angular frequency = ', angular_freq, ' microHz')
        print('Normalisation = ', normal, 'g*R_sun^2')
        print(f'H_{l} {n} {m}, {lprime} {nprime} {mprime} = ', gme, ' kG^2*R_Sun^3')
        print(f'H_{l} {n} {m}, {lprime} {nprime} {mprime}/ normal = ',
              gme / normal * 10 ** 12 * 10 ** 6 * mesa_data.R_sun, ' microHz^2')
        sme = gme/normal*10**12*10**6*mesa_data.R_sun-(angular_freq**2-(2*np.pi*frequencies_GYRE(l,n))**2)
        freq_shift_full = sme/(4*np.pi*angular_freq)
        print('Frequency shift with only SelfCoupling and complete quasi-degenerate perturbation Theory: ', freq_shift_full * 10 ** 3, ' nHz')
        # Kiefer and Roth 2018 approximation:
        # They included in GME normalization factor
        # Only SelfCoupling Terms
        # Works fine for eigenspaces with only SelfCoupling Terms (uncertainty compared to SelfCoupling from QDPT about 10^-6 nHz)
        print('\nKiefer and Roth 2018 approximation: ')
        approx_frequency_shift = (np.sqrt(angular_freq ** 2 + gme / normal * 10 ** 12 * 10 ** 6 * mesa_data.R_sun)
                                  - angular_freq) / (2 * np.pi)  # microHz, after Kiefer2018
        print(f'approx frequency shift Kiefer for l={l}, n={n}, m={m} =', approx_frequency_shift * 10 ** 3, 'nHz')
        print('Diff: approx - SelfCoupling QDPT = ', (freq_shift_full - approx_frequency_shift) * 10 ** 3, 'nHz')


if __name__ == '__main__':
    main()
