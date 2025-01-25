import numpy as np
import os
import GeneralMatrixElements_Parallel as GMEP
import Frequency_shift as fs
import radial_kernels
import time
from config import ConfigHandler


def calculate_freq_shift_quasi_degenerate(l, n, delta_freq_quadrat, magnetic_field_s, magnetic_field_sprime=None, eigentag=None, mesa_data=None, first_approx_data=None):
    try:
        start_time = time.time()

        if magnetic_field_sprime is None:
            magnetic_field_sprime = magnetic_field_s

        # Create eigenspace
        K_space = GMEP.eigenspace(l, n, delta_freq_quadrat, eigentag, first_approx_data)

        # Create and Save index map
        index_map = GMEP.create_index_map(l, n, K_space)
        GMEP.save_index_map_to_file(index_map, l, n, eigentag)

        # Calculate and save supermatrix
        # ALL GME involved are saved in database for later use
        sme = GMEP.supermatrix_parallel(l, n, magnetic_field_s, magnetic_field_sprime, K_space, mesa_data)
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

        # Calculate and save frequency shifts for (l,n)-modes
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

    # Load Model name
    model_name = config.get("ModelConfig", "model_name")

    # Create Output Directories
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
    temp_dir = os.path.join(gme_dir, 'Temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"Created directory: {temp_dir}")
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

    # Initialize the stellar model (MESA data)
    mesa_data = radial_kernels.MesaData(config=config)

    # Load the magnetic field parameters
    B_max = config.getfloat("MagneticFieldModel", "B_max")
    mu = config.getfloat("MagneticFieldModel", "mu")
    sigma = config.getfloat("MagneticFieldModel", "sigma")
    s = config.getint("MagneticFieldModel", "s")
    sprime = config.getint("MagneticFieldModel", "sprime", default=s)

    # Initialize the magnetic field model
    magnetic_field_s = radial_kernels.MagneticField(B_max=B_max, mu=mu, sigma=sigma, s=s, radius_array=mesa_data.radius_array)
    magnetic_field_sprime = radial_kernels.MagneticField(B_max=B_max, mu=mu, sigma=sigma, s=sprime, radius_array=mesa_data.radius_array)

    # Eigenspace width and eigentag
    delta_freq_quadrat = config.getfloat("Eigenspace", "delta_freq_quadrat")  #microHz^2
    eigentag = config.get("Eigenspace", "eigenspace_tag")

    # Compute lower turning points for first approximation
    if eigentag == 'FirstApprox':
        summary_dir = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', config.get("StellarModel", "summary_GYRE_path"))
        r_t = GMEP.lower_turning_points(summary_dir, mesa_data)
        first_app = GMEP.first_approx(r_t, magnetic_field_s)

    # Load the range for l and n
    l_min = config.getint("Range", "l_min")
    l_max = config.getint("Range", "l_max")
    n_min = config.getint("Range", "n_min")
    n_max = config.getint("Range", "n_max")

    # Compute frequency shifts for all multiplets in l, n parameterspace defined in [Range] given a specific eigenspace tag (Full, FirstApprox or SelfCoupling)
    for l in range(l_min, l_max+1):
        for n in range(n_min, n_max+1):
            if eigentag == 'FirstApprox':
                # Excludes multiplets which do not penetrate into effective magnetic field range from computation
                if (l, n) not in first_app['multiplets']:
                    continue

            if GMEP.frequencies_GYRE(l,n) is not None:
                print(f'Computing multiplet l={l}, n={n}, freq={GMEP.frequencies_GYRE(l,n)} microHz with eigenspace tag {eigentag}')
                calculate_freq_shift_quasi_degenerate(l, n, delta_freq_quadrat, magnetic_field_s, magnetic_field_sprime,
                                                      eigentag=eigentag, mesa_data=mesa_data,
                                                      first_approx_data=first_app if eigentag == 'FirstApprox' else None)

    print(f'All multiplets finished.')


if __name__ == '__main__':
    main()
