# General Matrix Element Computation

## Citation
If you use this code, please cite:
**Lindner and Roth (2025)**: *"A Revised Expression for the General Matrix Element of the Direct Effect of a Toroidal Magnetic Field on Stellar Oscillations"* (in progress).

## Contact
For any bugs, problems, or questions, please don't hesitate to contact me at: jlindner@tls-tautenburg.de

---

## Setup
### 1. Create a Virtual Environment
Ensure Python's virtual environment package is installed:
```bash
sudo apt install python3-venv -y
```

Create the environment (use the name `comput_main_env` if you intend to run the batch script; otherwise, adjust the path in the script accordingly):
```bash
python3 -m venv comput_main_env
```

Activate the environment:
```bash
source comput_main_env/bin/activate
```

Install dependencies (requirements file located in `__GME_ToroidalField__/GeneralMatrixElement/`):
```bash
pip install -r requirements.txt
```

### 2. Make the Batch Script Executable
```bash
chmod +x run_computation.sh
```

## Running Program

### 1. Set up `config.ini`
Set up `config.ini` according to your needs.

### 2. Run the Computation
You can run the computation in two ways:
- Using the batch script (recommended, as it activates the environment automatically and logs output):
  ```bash
  ./run_computation.sh
  ```
- Running the script directly:
  ```bash
  python3 Comput_main.py
  ```

---

## Configuration File (config.ini)

Before running the computation, you need to configure the `config.ini` file to match your setup. **For each new change (except for `[Range]` and `eigenspace_tag`) a new model name must be selected to avoid overwriting.**
The file consists of several sections:

### [ModelConfig]
- Defines the model name. 
- When making any changes (except for `[Range]` and `eigenspace_tag`), a new model name should be assigned.

### [StellarModel]
- Specifies the MESA model data, including the MESA LOG folder and paths to GYRE summary and detail data files.
- Eigenfunctions must be stored in a specific format (default of GYRE): `detail.l*.n+*.h5`.

### [MagneticFieldModel]
- Defines parameters for the Gaussian magnetic field model:
  - `B_max`: Maximum magnetic field strength (in kG).
  - `mu`: Location of the maximum magnetic field strength (in solar radii).
  - `sigma`: Standard deviation of the distribution (in solar radii).
  - `s`, `sprime`: Harmonic degrees of the magnetic field components.

### [Eigenspace]
- Controls eigenspace properties:
  - `delta_freq_quadrat`: Width of the eigenspace (in µHz²).
  - `eigenspace_tag`: Specifies the eigenspace type:
    - `Full`: Include all modes within the eigenspace width.
    - `FirstApprox`: Excludes modes with lower turning points larger than `r_thresh = mu + 3*sigma`, i.e., modes that do not propagate within the effective range of the magnetic field.
    - `SelfCoupling`: Includes only the reference multiplet in the eigenspace and does not allow cross-couplings.

### [Range]
- Defines the computation range:
  - Minimum and maximum harmonic degrees (`l_min`, `l_max`).
  - Minimum and maximum radial orders (`n_min`, `n_max`).

For more details, refer to the comments in `config.ini`.

---

## Plotting
To use plotting functions within the source code, activate the environment:
```bash
source comput_main_env/bin/activate
```

### `Plot_output.py`
- Generates various plots, including relative/absolute frequency shifts, 4D overview plots (x=m/l, y=l, z=freq_shift, color=n), frequency shift comparisons, frequency shifts overview plots as a function of frequency or lower turning point, and diagnostic plots for supermatrices.
- To enable plots:
  1. Modify the `main()` function by setting the appropriate boolean variables to `True`.
  2. Adjust the relevant parameters to suit your requirements.

**Note:** `plt.show()` may not work on some setups due to interactive plotting restrictions. To save the plots instead, set the `save` boolean to `True` in the plot initialization.

### `radial_kernels.py`
- Contains plots for density, magnetic field, radial kernels, and eigenfunctions (see development notes).

---

## Development Notes
- Various diagnostic tools are available in the `main()` function of different scripts, which can be enabled by setting the corresponding boolean variables to `True`.
- Some parameter inputs (e.g., magnetic field parameters and data paths) must be configured in `config.ini`.

### `GeneralMatrixElements_Parallel.py`
- `compare_eigenspace`: Compares eigenspaces using different eigentags (full eigenspace, first approximation, only self-coupling).
- `test_gme`: Computes a single generalised matrix element and records runtime.
- `investigate_hdf`: Investigates or merges HDF5 files.
- `search_eigenspace`: Searches for eigenspaces with modes in a given frequency interval (linear and squared differences).
- `lower_tp`: Filters lower turning points of a given multiplet.
- `test_kiefer`: Computes approximate frequency shifts based on Kiefer & Roth (2018) and compares the results with quasi-degenerate perturbation theory using only self-coupling terms.

### `radial_kernels.py`
- `plot_mesa_data`: Plots density derivatives over the star.
- `plot_magnetic_field`: Plots Gaussian radial profiles of the magnetic field distribution. Also creates a plot showing the radial and co-latitudinal distribution of the magnetic field.
- `test_radial_kernels`: Computes individual radial kernel values.
- `plot_kernel`: Plots radial kernels on logarithmic and linear scales.
- `compare`: Compares radial kernels for different multiplets. If split=True, creates two separate images, each containing four subplots.
- `plot_eigenfunctions`: Plots radial and horizontal eigenfunctions.

### `angular_kernels.py`
- `test_angular_kernels`: Computes individual angular kernel values.

### Database Structure
- GMEs are stored in HDF5 files. Each thread writes its own temporary HDF5 file, and these files are merged into the main HDF5 database after completing the computation of the supermatrix.
- All General Matrix Elements (GMEs) are indexed by `(l, n, m, l', n', m')`.
- Only non-zero GMEs are stored, ensuring coupling occurs only when selection rules are satisfied.

### Configuration File Handling
If using the config file outside `Comput_main.py`, initialize it explicitly:
```python
config = ConfigHandler("config.ini")
```
To access configuration parameters:
```python
config.get(section, option)  # Also available: getint(), getfloat()
```

**Note:** `ConfigHandler()` is implemented as a singleton, improving performance by caching the instance.

### Superpositions of Toroidal Magnetic Fields
- Computing superpositions requires significantly more computational time and postprocessing.
- For each combination of (s, s'), the program must be run separately for all required multiplets.
- Multiple databases are generated, each containing GMEs for the different combinations.
- GMEs with the same index `(l, n, m, l', n', m')` must be summed across these databases.
- The program can use the new GMEs under the assumption of completeness. The processed database must be stored in the model folder for superposed fields, allowing the program to fetch GMEs directly for frequency shift computations.
