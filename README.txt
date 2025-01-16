- If you use this code cite:
Lindner and Roth 2025: "A Revised Expression for the General Matrix Element of the Direct Effect of a Toroidal Magnetic Field on Stellar Oscillations"
(in Progress)


- In database all GMEs are indexed by (l,n,m,lprime,nprime,mprime) and only non-zero GMEs are stored, i.e. modes couple only (and have a non-zero GME) if both selection rules are satisfied.

- Diagnostic Plots for Supermatrix in GeneralMatrixElemements_Parallel.py are available, but paths to supermatrix_array file and save location have to be adjusted manually.

- If config file is used in the def main(): to test functions outside of the Comput_main.py, it has to be initialised once. ConfigHandler() was designed to be singleton, i.e. one instance is stored in Memory Cache and all other calls of ConfigHandler() access the open instance instead of reloading it. This improves Performance.
    # Initialize configuration handler
    config = ConfigHandler("config.ini")
    # Access config Parameter (get(), getint(), getfloat()):
    config.get(section, option)

- Their is still some testing code commented out in the main() functions of the Utility scripts, some of these codes still refer to older versions and do not work anymore, but they are still included for references

- In principle superpositions of toroidal magnetic fields can be computed. This requires significantly more Computational time and postprocessing. For each combination of s and sprime, the Programm has to run once (for all required multiplets). This will result into several db containing the GMEs of each combination. Now the GMEs with the same key-value (6 Indices: l, n, m, lprime, nprime, mprime) have to be added up. Finally, the Programm can be tricked to use the new GMEs assuming completeness by storing the db before processing in the new model folder for the superposed fields. The Programm will then fetch the GMEs directly from the db and computes the frequency shifts.