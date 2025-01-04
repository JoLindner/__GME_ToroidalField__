import numpy as np
import os
import GeneralMatrixElements_Parallel as GMEP
import Frequency_shift as fs
import radial_kernels
import time
from config import ConfigHandler

def calculate_freq_shift_quasi_degenerate(l,n,delta_freq_quadrat,magnetic_field_s, magnetic_field_sprime=None, eigentag=None):
    try:
        start_time = time.time()

        if magnetic_field_sprime is None:
            magnetic_field_sprime = magnetic_field_s

        #Create and Save index map
        index_map = GMEP.create_index_map(l,n,GMEP.eigenspace(l,n,delta_freq_quadrat, eigentag))
        GMEP.save_index_map_to_file(index_map, l, n, eigentag)

        #Calculate and save supermatrix
        #ALL GME involved are safed in database for later use
        sme = GMEP.supermatrix_parallel(l, n, delta_freq_quadrat, magnetic_field_s, magnetic_field_sprime, eigentag)
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
        np.savetxt(DATA_DIR, sme, delimiter=' ')
        print(f'Supermatrix of l={l} n={n} multiplet with {eigentag} eigenspace saved to {DATA_DIR}')

        #Calculate and save frequency shifts for (l,n)-modes
        f_shifts=fs.calculate_safe_extract_freq_shifts(l,n,eigentag)
        print(f'Results of Frequency shifts for l={l} n={n} multiplet:')
        print(f_shifts)

        end_time = time.time()
        print("Elapsed time:", end_time - start_time, "seconds")
        print('Finished.\n\n\n')

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error during combining the results: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in calculate_freq_shift_quasi_degenerate: {e}")



def main():
    # Initialize configuration handler
    # Use NEW model_name for ALL parameter changes (except eigenspace_tag and [Range] settings)
    config = ConfigHandler("config.ini")

    #Load Model name
    model_name = config.get("ModelConfig", "model_name")

    #Create Directories
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    model_output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        print(f"Created directory: {model_output_dir}")
    gme_dir = os.path.join(model_output_dir, 'GeneralMatrixElements')
    if not os.path.exists(gme_dir):
        os.makedirs(gme_dir)
        print(f"Created directory: {gme_dir}")
    sm_dir = os.path.join(model_output_dir, 'Supermatrices')
    if not os.path.exists(sm_dir):
        os.makedirs(sm_dir)
        print(f"Created directory: {sm_dir}")
    index_map_dir = os.path.join(sm_dir, 'IndexMaps')
    if not os.path.exists(index_map_dir):
        os.makedirs(index_map_dir)
        print(f"Created directory: {index_map_dir}")
    frequency_shifts_dir = os.path.join(model_output_dir, 'FrequencyShifts')
    if not os.path.exists(frequency_shifts_dir):
        os.makedirs(frequency_shifts_dir)
        print(f"Created directory: {frequency_shifts_dir}")


    # Load the magnetic field parameters
    B_max = config.getfloat("MagneticFieldModel", "B_max")
    mu = config.getfloat("MagneticFieldModel", "mu")
    sigma = config.getfloat("MagneticFieldModel", "sigma")
    s = config.getfloat("MagneticFieldModel", "s")

    # Initialize the magnetic field model
    magnetic_field_s = radial_kernels.MagneticField(B_max=B_max, mu=mu, sigma=sigma, s=s)


    #eigenspace width and eigentag
    delta_freq_quadrat = config.getfloat("Eigenspace", "delta_freq_quadrat")  #microHz^2
    eigentag = config.get("Eigenspace", "eigenspace_tag")

    # Load the range for l and n
    l_min = config.getint("Range", "l_min")
    l_max = config.getint("Range", "l_max")
    n_min = config.getint("Range", "n_min")
    n_max = config.getint("Range", "n_max")


    # Compute frequency shifts for all multiplets in l, n parameterspace defined in [Range] given a specific eigenspace tag (Full, FirstApprox or SelfCoupling)
    for l in range(l_min, l_max+1):
        for n in range(n_min, n_max+1):
            # Excludes multiplets which do not penetrate into effective magnetic field range
            criterium_large = (128 <= l < 138 and n >= 13) or (
                    118 <= l < 128 and n >= 12) or (
                                      108 <= l < 118 and n >= 11) or (
                                      98 <= l < 107 and n >= 10) \
                              or (88 <= l < 98 and n >= 9) or (
                                      79 <= l < 88 and n >= 8) or (
                                      69 <= l < 79 and n >= 7) \
                              or (60 <= l < 69 and n >= 6) or (
                                      51 <= l < 60 and n >= 5) or (
                                      42 <= l < 51 and n >= 4) \
                              or (31 <= l < 42 and n >= 3) or (
                                      20 <= l < 31 and n >= 2) or l <= 19 and n >= 1
            if GMEP.frequencies_GYRE(l,n) is not None and criterium_large:
                print(f'Computing multiplet l={l}, n={n}, freq={GMEP.frequencies_GYRE(l,n)} microHz with eigenspace tag {eigentag}')
                calculate_freq_shift_quasi_degenerate(l,n,delta_freq_quadrat, magnetic_field_s, eigentag=eigentag)

    print(f'All multiplets finished.')


    '''
    #For master thesis
    l = 5
    n = [6,12,18]

    for value in n:
        print(f'Computing l={l}, n={value}, freq={GMEP.frequencies_GYRE(l,value)} microHz')
        calculate_freq_shift_quasi_degenerate(l,n,delta_freq_quadrat, magnetic_field_s, eigentag=eigentag)
    '''

if __name__== '__main__':
    main()
