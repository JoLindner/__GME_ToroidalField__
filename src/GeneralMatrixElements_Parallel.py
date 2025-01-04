import numpy as np
import angular_kernels
import radial_kernels
import os
import pygyre as pg
import concurrent.futures
import time
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.colors import ListedColormap
from config import ConfigHandler
import duckdb


def single_GM(l,n,m,lprime,nprime,mprime,magnetic_field_s, magnetic_field_sprime=None, model_name=None):
    #Output directory initialization
    db_name_string = f'gme_results_{model_name}.db'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', model_name, 'GeneralMatrixElements', db_name_string)

    # Check if the database file already exists
    if not os.path.exists(DATA_DIR):
        print(f"Creating new database: {db_name_string}")

    try:
        conn = duckdb.connect(database=DATA_DIR)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS gme_results (
                                l INTEGER,
                                n INTEGER,
                                m INTEGER,
                                lprime INTEGER,
                                nprime INTEGER,
                                mprime INTEGER,
                                result DOUBLE,
                                PRIMARY KEY (l, n, m, lprime, nprime, mprime)
                              )''')

        # Check if the result already exists
        cursor.execute('''SELECT result FROM gme_results WHERE l=? AND n=? AND m=? AND lprime=? AND nprime=? AND mprime=?''',
                           (l, n, m, lprime, nprime, mprime))
        result_row = cursor.fetchone()

        #Return the GME if it already exists
        if result_row is not None:
            return result_row[0]

        #Magnetic fields
        if magnetic_field_sprime == None:
            magnetic_field_sprime = magnetic_field_s
        s = magnetic_field_s.s
        sprime = magnetic_field_sprime.s

        #First selection rule
        if m != mprime:
            return 0.0

        #Second selection rule
        if (l+lprime+s+sprime) % 2 == 1:
            return 0.0

        #GME
        radius_array = radial_kernels.MESA_structural_data()[0]
        R_args = (l, n, lprime, nprime, radius_array, magnetic_field_s, magnetic_field_sprime)
        S_args = (lprime, l, s, sprime, mprime, m)

        #H_k',k
        general_matrix_element=1/(4*np.pi)*(radial_kernels.R1(*R_args)[0]*angular_kernels.S1(*S_args)\
                                                +radial_kernels.R2(*R_args)[0]*(angular_kernels.S2(*S_args)-angular_kernels.S5(*S_args))\
                                                -radial_kernels.R3(*R_args)[0]*(angular_kernels.S3(*S_args)+angular_kernels.S6(*S_args))\
                                                +radial_kernels.R4(*R_args)[0]*angular_kernels.S4(*S_args)\
                                                +radial_kernels.R5(*R_args)[0]*(angular_kernels.S7(*S_args)+angular_kernels.S8(*S_args))\
                                                +radial_kernels.R6(*R_args)[0]*(angular_kernels.S9(*S_args)-angular_kernels.S10(*S_args)+angular_kernels.S11(*S_args)\
                                                -2*angular_kernels.S13(*S_args)+angular_kernels.S14(*S_args)-angular_kernels.S15(*S_args)-angular_kernels.S16(*S_args)\
                                                -angular_kernels.S18(*S_args)-angular_kernels.S19(*S_args)-angular_kernels.S20(*S_args)-angular_kernels.S22(*S_args)\
                                                -angular_kernels.S23(*S_args))\
                                                -radial_kernels.R7(*R_args)[0]*angular_kernels.S17(*S_args)\
                                                -radial_kernels.R8(*R_args)[0]*angular_kernels.S21(*S_args))

        if general_matrix_element is None:
            raise ValueError(f"The computed GME is None for the input parameters: l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}.")

        #SAVES ONLY modes that fulfill the selection rules, i.e. are non-zero, INTO DB
        cursor.execute('''INSERT INTO gme_results (l, n, m, lprime, nprime, mprime, result) 
                              VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (l, n, m, lprime, nprime, mprime, general_matrix_element))
        conn.commit()
        print(f'Saved GM: l={l}, n={n}, m={m}, lprime={lprime}, nprime={nprime}, mprime={mprime}, result={general_matrix_element}')
        return general_matrix_element

    except duckdb.Error as e:
        # Catch database errors and raise appropriate messages
        print(f"Database error occurred: {e}")
        conn.rollback()  # In case of failure, rollback any changes made during the transaction
        raise e  # Re-raise the error to allow the calling function to handle it
    except Exception as e:
        # Catch other errors (e.g., programming errors, valueError)
        print(f"An error occurred: {e}")
        raise e  # Re-raise to propagate the error

    finally:
        if conn:
            conn.close()

def normalization(l,n):
    #Get MESA structural data
    radius_array = radial_kernels.MESA_structural_data()[0]
    r_sun = radial_kernels.MESA_structural_data()[3]  # r_sun in cm
    rho_0 = radial_kernels.MESA_structural_data()[2]*r_sun**3 #g/R_sun^3
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

def eigenspace(l,n, delta_freq_quadrat, eigen_tag=None):
    try:
        #TAGS: Default/Full: full eigenspace; FirstApprox: first approximation; SelfCoupling: Only self-coupling (second Approximation)

        # doesnt matter here if I calculate the eigenspaces from the frequencies insteat of angular frequencies
        freq_ref = frequencies_GYRE(l, n)

        # Initialize configuration handler
        config = ConfigHandler()

        # Read path to summary file
        summary_GYRE_path = config.get("StellarModel", "summary_GYRE_path")
        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', summary_GYRE_path)
        summary_file = pg.read_output(DATA_DIR)
        K_space = []

        if eigen_tag == None or eigen_tag=='Full':
            #FULL EIGENSPACE
            #delta_freq in microHz
            for row in summary_file:
                if abs(row['freq'].real**2-freq_ref**2) <= delta_freq_quadrat and row['l'] > 1:   #exclude l>1, since eq. A.19 breaks down for these cases
                    new_row = {
                        'freq': row['freq'],
                        'n': row['n_pg'],
                        'l': row['l']}
                    K_space.append(new_row)

        elif eigen_tag == 'FirstApprox':
            # Apply first approximation
            for row in summary_file:
                criterium_large = (128 <= row['l'] < 138 and row['n_pg'] >= 13) or (
                            118 <= row['l'] < 128 and row['n_pg'] >= 12) or (
                                              108 <= row['l'] < 118 and row['n_pg'] >= 11) or (
                                              98 <= row['l'] < 107 and row['n_pg'] >= 10) \
                                  or (88 <= row['l'] < 98 and row['n_pg'] >= 9) or (
                                              79 <= row['l'] < 88 and row['n_pg'] >= 8) or (
                                              69 <= row['l'] < 79 and row['n_pg'] >= 7) \
                                  or (60 <= row['l'] < 69 and row['n_pg'] >= 6) or (
                                              51 <= row['l'] < 60 and row['n_pg'] >= 5) or (
                                              42 <= row['l'] < 51 and row['n_pg'] >= 4) \
                                  or (31 <= row['l'] < 42 and row['n_pg'] >= 3) or (
                                              20 <= row['l'] < 31 and row['n_pg'] >= 2) or row['l'] <= 19 and row[
                                      'n_pg'] >= 1
                #criterium_small = row['l'] < 138

                if abs(row['freq'].real ** 2 - freq_ref ** 2) <= delta_freq_quadrat and row['l'] > 1:
                    if criterium_large:
                        new_row = {
                            'freq': row['freq'],
                            'n': row['n_pg'],
                            'l': row['l']}
                        K_space.append(new_row)

        elif eigen_tag == 'SelfCoupling':
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


def eigenspace_mode_search(l,n, freq_interval):
    #Search quasi-degenerate multiplets with frequencies closer than 0.1 microHz to the reference multiplet

    #doesnt matter here if I calculate the eigenspaces from the frequencies insteat of angular frequencies
    freq_ref = frequencies_GYRE(l,n)

    # Initialize configuration handler
    config = ConfigHandler()

    # Read path to summary file
    summary_GYRE_path = config.get("StellarModel", "summary_GYRE_path")
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', summary_GYRE_path)
    summary_file = pg.read_output(DATA_DIR)
    K_space=[]

    for row in summary_file:
        criterium_large=(128<=row['l']<138 and row['n_pg']>=13) or (118<=row['l']<128 and row['n_pg']>=12) or (108<=row['l']<118 and row['n_pg']>=11) or (98<=row['l']<107 and row['n_pg']>=10) \
                  or (88 <= row['l'] < 98 and row['n_pg'] >= 9) or (79 <= row['l'] < 88 and row['n_pg'] >= 8) or (69<= row['l'] < 79 and row['n_pg'] >= 7) \
                  or (60 <= row['l'] < 69 and row['n_pg'] >= 6) or (51 <= row['l'] < 60 and row['n_pg'] >= 5) or (42 <= row['l'] < 51 and row['n_pg'] >= 4) \
                  or (31 <= row['l'] < 42 and row['n_pg'] >= 3) or (20 <= row['l'] < 31 and row['n_pg'] >= 2) or row['l']<19 and row['n_pg']>=1
        if abs(row['freq'].real-freq_ref) <= freq_interval:
            if criterium_large==True:
                new_row = {
                    'freq': row['freq'],
                    'n': row['n_pg'],
                    'l': row['l']}
                K_space.append(new_row)

    return K_space


def supermatrix_element(omega_ref,l,n,m,lprime,nprime,mprime, magnetic_field_s, magnetic_field_sprime=None, model_name=None):
    try:
        if magnetic_field_sprime == None:
            magnetic_field_sprime = magnetic_field_s

        #existence of gme and normal is checked in the functions
        gme = single_GM(l,n,m,lprime,nprime,mprime,magnetic_field_s,magnetic_field_sprime,model_name)
        normal = normalization(l,n)

        #delta function
        delta = 1 if l == lprime and n == nprime and m == mprime else 0

        #includes conversion factor 10**12*10**6*R_sun to yield microHz^2
        sme = gme/normal*10**12*10**6*radial_kernels.MESA_structural_data()[3]-(omega_ref**2-(2*np.pi*frequencies_GYRE(l,n))**2)*delta  #microHz^2

        if sme is None:
            raise ValueError(f"Supermatrix element (SME) is None for inputs: "
                             f"l={l}, n={n}, m={m}, l'={lprime}, n'={nprime}, m'={mprime}.")

        return sme, gme, normal

    except Exception as e:
        print(f"An error occurred in computing the 'supermatrix_element': {e}")
        raise e


def supermatrix_parallel_one_row(row,l,n,delta_freq_quadrat,magnetic_field_s, magnetic_field_sprime=None, eigen_tag=None, model_name=None):
    try:
        if magnetic_field_sprime == None:
            magnetic_field_sprime = magnetic_field_s
        kprime = row[0]
        mprime = row[1]
        omega_ref = 2*np.pi*frequencies_GYRE(l,n)
        K_space = eigenspace(l,n, delta_freq_quadrat, eigen_tag)

        size=0
        for i in range(0,len(K_space)):
            for iprime in range(0, len(K_space)):
                size=size+(2*K_space[i]['l']+1)*(2*K_space[iprime]['l']+1)
        if np.sqrt(size) %1 ==0:
            matrix_size=int(np.sqrt(size))
        else:
            raise ValueError('Matrixsize not an integer')

        supermatrix_array_row = np.empty((matrix_size), dtype=np.float64)

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
                sme = supermatrix_element(omega_ref,l,n,m,lprime,nprime,mprime,magnetic_field_s, magnetic_field_sprime, model_name)[0]
                supermatrix_array_row[col] = sme
                col += 1
        return np.transpose(supermatrix_array_row)

    except Exception as e:
        print(f"An error occurred in computing a row of the supermatrix: {e}")
        raise e

def supermatrix_parallel(l,n, delta_freq_quadrat, magnetic_field_s, magnetic_field_sprime=None, eigen_tag=None):
    # Initialize configuration handler
    config = ConfigHandler()
    #Load Model name
    model_name = config.get("ModelConfig", "model_name")
    #Check if Model name exists
    if not model_name:
        raise ValueError("The 'model_name' could not be fetched from the config.ini file under [ModelConfig] section.")

    try:
        if magnetic_field_sprime == None:
            magnetic_field_sprime = magnetic_field_s

        #generate eigenspace
        K_space = eigenspace(l,n,delta_freq_quadrat, eigen_tag)

        #Create row indices
        rows = []
        for kprime in range(0, len(K_space)):
            l_aux = K_space[kprime]['l']
            for mprime in range(0, 2*l_aux+1):
                rows.append([kprime,mprime])

        # gets either SLURM_CPUS_PER_TASK or the number of CPU cores from the system
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))  #number of cpus

        # Execute parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(supermatrix_parallel_one_row, rows, [l] * len(rows), [n] * len(rows),
                                       [delta_freq_quadrat] * len(rows), [magnetic_field_s] * len(rows), [magnetic_field_sprime] * len(rows), [eigen_tag] * len(rows), [model_name] * len(rows)))

        # Combine rows
        combined_result = np.vstack(results)
        return combined_result

    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        raise e

#INDEX MAP
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


        if eigentag == None or eigentag=='Full':
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
        # Here only used for creating supermatrix plots
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



#SUPERMATRIX PLOTS
def plot_supermatrix_var(l,n, linthresh):
    #load supermatrix
    name_string = f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')
    supermatrix_array = supermatrix_array.astype(float)

    # Load index_map
    name_string = f'index_map_supermatrix_array_{l}_{n}.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    index_map = load_index_map_from_file(data_dir)

    #linthresh = np.min(np.abs(supermatrix_array[supermatrix_array != 0]))
    vmax = np.max(np.abs(supermatrix_array))
    norm = SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, linscale=2)
    fig, ax = plt.subplots(figsize=(10,9))
    cax = ax.pcolormesh(supermatrix_array, norm=norm,  cmap='seismic', alpha=1)
    # Add colorbar
    cb = plt.colorbar(cax, ax=ax, label=f'Supermatrixelement in $\mu$Hz$^2$', pad=0.02)
    cb.set_label(label=f'Supermatrixelement in $\mu$Hz$^2$',fontsize=16)
    cb.ax.tick_params(labelsize=14)
    #plt.xlabel('l, n')
    #plt.ylabel('lprime, nprime')

    major_ticks = []
    minor_ticks = []
    tickslabel = []
    minor_tickslabel = []
    for i in range(len(index_map[0])):
        column=index_map[0][i]
        minor_ticks.append(i+0.5)
        minor_tickslabel.append(f'm={column["m"]}')
        l=column['l']
        if l==abs(column['m']):
            major_ticks.append(i+0.5)
            tickslabel.append(f'l={l}, n={column["n"]}, m={column["m"]}')


    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    for minor_tick in ax.xaxis.get_minorticklocs():
        ax.text(minor_tick, ax.get_ylim()[0] - 1, minor_tickslabel[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='top', rotation=90)
    for minor_tick in ax.yaxis.get_minorticklocs():
        ax.text(ax.get_xlim()[0] - 1, minor_tick, minor_tickslabel[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='top')

    ax.set_xticklabels(tickslabel, rotation=90, fontsize=12)
    ax.set_yticklabels(tickslabel, fontsize=12)

    plt.title(f'Supermatrix for multiplet around l={l}, n={n}', fontsize=16)
    plt.savefig(f'Images/Supermatrix_Pixel_Plot_l={l}_n={n}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_supermatrix(l,n, linthresh, trunc=None):
    #load supermatrix
    name_string = f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')
    supermatrix_array = supermatrix_array.astype(float)

    # Load index_map
    name_string = f'index_map_supermatrix_array_{l}_{n}.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    index_map = load_index_map_from_file(data_dir)

    if trunc is not None:
        index_map_l = np.unique(index_map['l'])[:-trunc]
        mask = np.isin(index_map['l'], index_map_l) & np.isin(index_map['lprime'], index_map_l)
        true_indices = np.argwhere(mask[0]==False)
        index_map = np.delete(index_map, true_indices, axis=0)
        index_map = np.delete(index_map, true_indices, axis=1)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=0)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=1)

    #print elements above threshold:
    thresh=linthresh
    data_above=[]
    for i in range(len(supermatrix_array)):
        for j in range(len(supermatrix_array[i])):
            if np.abs(supermatrix_array[i][j])>thresh:
                data_above.append((supermatrix_array[i][j], index_map[i][j]))
    sorted_data_above = sorted(data_above, key=lambda x: x[1][0])
    for value, index in sorted_data_above:
        print(value, index)
    
    #linthresh = np.min(np.abs(supermatrix_array[supermatrix_array != 0]))
    #print(linthresh)
    vmax = np.max(np.abs(supermatrix_array))
    norm = SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, linscale=2)
    fig, ax = plt.subplots(figsize=(16,16))
    cax = ax.pcolormesh(supermatrix_array, norm=norm,  cmap='seismic', alpha=1)
    # Add colorbar
    cb = plt.colorbar(cax, ax=ax, label=f'Supermatrixelement in µHz$^2$', pad=0.02, shrink = 0.84)
    cb.set_label(label=f'Supermatrixelement in µHz$^2$',fontsize=16)
    cb.ax.tick_params(labelsize=14)
    plt.title(f'Supermatrix for quasi-degenerate multiplet around l={l}, n={n}', fontsize=16)

    major_ticks = []
    minor_ticks = []
    tickslabel_x = []
    tickslabel_y = []
    minor_tickslabel_x = []
    minor_tickslabel_y = []
    for i in range(len(index_map[0])):
        column=index_map[0][i]
        l_co=column['l']
        #tick only every x-th m and m=0
        every_x_tick=2
        if abs(column['m']) %every_x_tick == 0:
            minor_ticks.append(i+0.5)
            minor_tickslabel_x.append(f'm={column["m"]}')
            minor_tickslabel_y.append(f'm\'={column["m"]}')

        if l_co==-column['m']:
            major_ticks.append(i)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')
        elif i==len(index_map[0])-1:
            major_ticks.append(i+1)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')

    for i in range(len(major_ticks) - 1):
        x_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        #displaces first label
        if i == 0 and trunc is not None:
            ax.text(x_mid, ax.get_ylim()[0] - 0.090 * (ax.get_ylim()[1] - ax.get_ylim()[0]), tickslabel_x[i],
                    ha='center', va='center', fontsize=16)  # adjust factor for y position
        else:
            ax.text(x_mid, ax.get_ylim()[0]-0.065*(ax.get_ylim()[1]-ax.get_ylim()[0]), tickslabel_x[i], ha='center', va='center', fontsize=16) #adjust factor for y position

    for i in range(len(major_ticks) - 1):
        y_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        if i==0 and trunc is not None:
            ax.text(ax.get_xlim()[0] - 0.090 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_mid, tickslabel_y[i],
                    ha='center', va='center', fontsize=16, rotation=90)  # adjust factor for y position
        else:
            ax.text(ax.get_xlim()[0]-0.065*(ax.get_xlim()[1]-ax.get_xlim()[0]), y_mid, tickslabel_y[i], ha='center', va='center', fontsize=16, rotation=90) #adjust factor for y position

    #grid
    for xline in major_ticks:
        ax.vlines(xline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='gray', linestyle='--', linewidth=1, zorder=1, alpha=0.9)
    for yline in major_ticks:
        ax.hlines(yline, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='gray', linestyle='--', linewidth=1, zorder=1, alpha=0.9)

    ax.set_aspect(aspect='equal')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.tick_params('x', length=50)
    ax.tick_params('y', length=50)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)

    displacement_factor = 2
    for minor_tick in ax.xaxis.get_minorticklocs():
        ax.text(minor_tick, ax.get_ylim()[0] - displacement_factor, minor_tickslabel_x[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='center', rotation=90)
    for minor_tick in ax.yaxis.get_minorticklocs():
        ax.text(ax.get_xlim()[0] - displacement_factor, minor_tick, minor_tickslabel_y[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='center')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.draw()
    plt.savefig(f'Images/Supermatrix_Pixel_Plot_l={l}_n={n}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_supermatrix_l5_n12(l, n, linthresh, trunc=None):
    # load supermatrix
    name_string = f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')
    supermatrix_array = supermatrix_array.astype(float)

    # Load index_map
    name_string = f'index_map_supermatrix_array_{l}_{n}.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices',
                            name_string)
    index_map = load_index_map_from_file(data_dir)

    if trunc is not None:
        index_map_l = np.unique(index_map['l'])[:-trunc]
        mask = np.isin(index_map['l'], index_map_l) & np.isin(index_map['lprime'], index_map_l)
        true_indices = np.argwhere(mask[0] == False)
        index_map = np.delete(index_map, true_indices, axis=0)
        index_map = np.delete(index_map, true_indices, axis=1)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=0)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=1)

    # print elements above threshold:
    thresh = linthresh
    data_above = []
    for i in range(len(supermatrix_array)):
        for j in range(len(supermatrix_array[i])):
            if np.abs(supermatrix_array[i][j]) > thresh:
                data_above.append((supermatrix_array[i][j], index_map[i][j]))
    sorted_data_above = sorted(data_above, key=lambda x: x[1][0])
    for value, index in sorted_data_above:
        print(value, index)

    # linthresh = np.min(np.abs(supermatrix_array[supermatrix_array != 0]))
    # print(linthresh)
    vmax = np.max(np.abs(supermatrix_array))
    #norm = SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, linscale=2)
    fig, ax = plt.subplots(figsize=(13, 13))
    #Slice seismic colorbar
    seismic = plt.get_cmap("seismic")
    red_half = seismic(np.linspace(0.5, 1, 1024))
    red_cmap = ListedColormap(red_half)

    cax = ax.pcolormesh(supermatrix_array, cmap=red_cmap, alpha=1)
    # Add colorbar
    cb = plt.colorbar(cax, ax=ax, label=f'Supermatrixelement in µHz$^2$', pad=0.02, shrink=0.84)
    cb.set_label(label=f'Supermatrixelement in µHz$^2$', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    plt.title(f'Supermatrix for quasi-degenerate multiplet around l={l}, n={n}', fontsize=16)

    major_ticks = []
    minor_ticks = []
    tickslabel_x = []
    tickslabel_y = []
    minor_tickslabel_x = []
    minor_tickslabel_y = []
    for i in range(len(index_map[0])):
        column = index_map[0][i]
        l_co = column['l']
        minor_ticks.append(i + 0.5)
        minor_tickslabel_x.append(f'm={column["m"]}')
        minor_tickslabel_y.append(f'm\'={column["m"]}')

        if l_co == -column['m']:
            major_ticks.append(i)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')
        elif i == len(index_map[0]) - 1:
            major_ticks.append(i + 1)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')

    for i in range(len(major_ticks) - 1):
        x_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        ax.text(x_mid, ax.get_ylim()[0] - 0.075 * (ax.get_ylim()[1] - ax.get_ylim()[0]), tickslabel_x[i],
                    ha='center', va='center', fontsize=16)  # adjust factor for y position

    for i in range(len(major_ticks) - 1):
        y_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        ax.text(ax.get_xlim()[0] - 0.075 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_mid, tickslabel_y[i],
                    ha='center', va='center', fontsize=16, rotation=90)  # adjust factor for y position

    # grid
    for xline in major_ticks:
        ax.vlines(xline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='gray', linestyle='--', linewidth=1,
                  zorder=1, alpha=0.9)
    for yline in major_ticks:
        ax.hlines(yline, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='gray', linestyle='--', linewidth=1,
                  zorder=1, alpha=0.9)

    ax.set_aspect(aspect='equal')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.tick_params('x', length=50)
    ax.tick_params('y', length=50)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)

    displacement_factor = 0.35
    for minor_tick in ax.xaxis.get_minorticklocs():
        ax.text(minor_tick, ax.get_ylim()[0] - displacement_factor, minor_tickslabel_x[minor_ticks.index(minor_tick)],
                fontsize=10, color='gray', ha='center', va='center', rotation=90)
    for minor_tick in ax.yaxis.get_minorticklocs():
        ax.text(ax.get_xlim()[0] - displacement_factor, minor_tick, minor_tickslabel_y[minor_ticks.index(minor_tick)],
                fontsize=10, color='gray', ha='center', va='center')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.draw()
    plt.savefig(f'Images/Supermatrix_Pixel_Plot_l={l}_n={n}_linear.png', dpi=300, bbox_inches='tight')
    plt.show()



#TEST AREA
def main():
    # Initialize configuration handler
    config = ConfigHandler("config.ini")

    #Eigenspace
    delta_freq_quadrat = config.getfloat("Eigenspace", "delta_freq_quadrat")  #microHz^2
    eigen_tag = config.get("Eigenspace", "eigenspace_tag")

    # Omega_ref
    l = 2
    n = 3
    linthresh=1e-16
    freq = frequencies_GYRE(l, n)
    print(freq)
    # Eigenspace
    K_space_old = eigenspace(l, n, delta_freq_quadrat)
    print('old: ',len(K_space_old), K_space_old)

    K_space_1approx = eigenspace(l, n, delta_freq_quadrat, eigen_tag='FirstApprox')
    print('new: ',len(K_space_1approx), K_space_1approx)

    K_space_2approx = eigenspace(l, n, delta_freq_quadrat, 'SelfCoupling')
    print('new: ',len(K_space_2approx), K_space_2approx)

    #plot_supermatrix(l, n, linthresh,3)


    '''
    #Eigenspace mode search
    #searches for eigenspaces containing 2 or more modes in the first approximation, eigenspace width is defined either by a frequency interval or delta_freq_quadrat
    n_len2, n_len2_quadrat = 0,0
    n_len3,n_len3_quadrat = 0,0
    n_len_larger, n_len_larger_quadrat = 0,0
    for l in range(0,150):
        for n in range(0,36):
            try:
                K_space=eigenspace_mode_search(l,n, 0.1)
                K_space_quadrat = eigenspace(l, n, delta_freq_quadrat, eigen_tag='FirstApprox')
                if len(K_space)>1:
                    if len(K_space)==2:
                        n_len2+=1
                        #print('K',K_space)
                    elif len(K_space)==3:
                        n_len3+=1
                        #print('K',K_space)
                    else:
                        n_len_larger+=1
                        print('K',K_space)

                if len(K_space_quadrat)>1:
                    if len(K_space_quadrat)==2:
                        n_len2_quadrat+=1
                        #print('K_qua',K_space_quadrat)
                    elif len(K_space_quadrat)==3:
                        n_len3_quadrat+=1
                        #print('K_qua',K_space_quadrat)
                    else:
                        n_len_larger_quadrat+=1
                        print('K_qua',K_space_quadrat)

            except:
                continue
    print('Eigenspace with length 2: ',n_len2)
    print('Eigenspace with length 3: ',n_len3)
    print('Eigenspace with length larger 3: ',n_len_larger)
    print('Eigenspace quadrat with length 2: ',n_len2_quadrat)
    print('Eigenspace quadrat with length 3: ',n_len3_quadrat)
    print('Eigenspace quadrat with length larger 3: ',n_len_larger_quadrat)
    '''




    '''
    #TEST KIEFER APPROX
    #initialize magnetic field
    magnetic_field_modelA = radial_kernels.MagneticField(B_max=300, mu=0.713, sigma=0.05, s=2)

    m=1
    lprime=l
    nprime=n
    mprime=1
    angular_freq=2*np.pi*frequencies_GYRE(l,n)  #microHz
    gme = single_GM(l,n,m,lprime,nprime,mprime,magnetic_field_modelA)
    print(gme)
    normal = normalization(l, n)
    print('Normalisation =', normal, 'g*R_sun^2')
    print(f'H_{l} {n} {m},{lprime} {nprime} {mprime} =', gme, ' kG^2*R_Sun^3')
    print(f'H_{l} {n} {m},{lprime} {nprime} {mprime}/ normal =',
          gme / normal * 10 ** 12 * 10 ** 6 * radial_kernels.MESA_structural_data()[3], ' microHz^2')
    print(np.sqrt(angular_freq ** 2 + gme / normal * 10 ** 12 * 10 ** 6 * radial_kernels.MESA_structural_data()[3]), 'microHz')
    # solution for l=lprime=2, n=5, nprime=4, m=mprime=1:  H_k,k'=-2259.118191997141 kG^2*R_Sun^3
    approx_frequency_shift = (np.sqrt(
        angular_freq ** 2 + gme / normal * 10 ** 12 * 10 ** 6 * radial_kernels.MESA_structural_data()[3]) - angular_freq) / (
                                         2 * np.pi)  # microHz, after Kiefer2018
    print(f'approx frequency shift =', approx_frequency_shift, 'microHz')
    print(f'approx frequency shift =', approx_frequency_shift * 10 ** 3, 'nHz')
    '''
    '''
    #Delta freq^2 = 30 microHz^2
    delta_freq_quadrat = 10000  #microHz^2

    #Omega_ref
    l=5
    n=5
    freq=frequencies_GYRE(l,n)
    print(freq)
    #Eigenspace
    K_space=eigenspace(l,n,delta_freq_quadrat)
    print(len(K_space),K_space)

    #initialize magnetic field
    magnetic_field_modelA = radial_kernels.MagneticField(B_max=50, mu=0.7, sigma=0.04, s=2)

    m=1
    lprime=5
    nprime=5
    mprime=1
    freq=frequencies_GYRE(l,n)  #microHz
    gme = single_GM(l,n,m,lprime,nprime,mprime,magnetic_field_modelA)
    print(gme)
    normal = normalization(l, n)
    print('Normalisation =', normal, 'g*R_sun^2')
    print(f'H_{l} {n} {m},{lprime} {nprime} {mprime} =', gme, ' kG^2*R_Sun^3')
    print(f'H_{l} {n} {m},{lprime} {nprime} {mprime}/ normal =',
          gme / normal * 10 ** 12 * 10 ** 6 * radial_kernels.MESA_structural_data()[3], ' microHz^2')
    print(np.sqrt(freq ** 2 + gme / normal * 10 ** 12 * 10 ** 6 * radial_kernels.MESA_structural_data()[3]), 'microHz')
    # solution for l=lprime=2, n=5, nprime=4, m=mprime=1:  H_k,k'=-2259.118191997141 kG^2*R_Sun^3
    approx_frequency_shift = (np.sqrt(
        freq ** 2 + gme / normal * 10 ** 12 * 10 ** 6 * radial_kernels.MESA_structural_data()[3]) - freq) / (
                                         2 * np.pi)  # microHz, after Kiefer2018
    print(f'approx frequency shift =', approx_frequency_shift, 'microHz')
    print(f'approx frequency shift =', approx_frequency_shift * 10 ** 3, 'nHz')
    #

    #Delta freq^2 = 10^4 microHz^2
    delta_freq_quadrat = 10000 #microHz^2

    #Omega_ref
    l=5
    n=5
    #freq=frequencies_GYRE(l,n)
    #Eigenspace
    #K_space=eigenspace(l,n,delta_freq_quadrat)
    '''
    '''
    #Supermatrix
    start_time = time.time()
    index_map=create_index_map(l,n,eigenspace(l,n,delta_freq_quadrat))
    save_index_map_to_file(index_map, l,n)
    #sme = supermatrix_parallel(l, n, delta_freq_quadrat, magnetic_field_modelA)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time parallel:", elapsed_time, "seconds")
    name_string=f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)
    #np.savetxt(DATA_DIR, sme, delimiter=' ')
    '''


if __name__== '__main__':
    main()
