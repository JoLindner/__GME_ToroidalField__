import numpy as np
import os
import GeneralMatrixElements_Parallel as GMEP
import Frequency_shift as fs
import Frequency_shift_first_approx as fs_1a
import Frequency_shift_second_approx as fs_2a
import radial_kernels
import time

def calculate_freq_shift_quasi_degenerate(l,n,delta_freq_quadrat,magnetic_field_s, magnetic_field_sprime=None):
    if magnetic_field_sprime == None:
        magnetic_field_sprime = magnetic_field_s

    start_time = time.time()
    #Create and Save index map
    index_map = GMEP.create_index_map(l,n,GMEP.eigenspace(l,n,delta_freq_quadrat))
    GMEP.save_index_map_to_file(index_map, l,n)

    #Calculate and save supermatrix
    #ALL GME involved are safed as well for later use
    sme = GMEP.supermatrix_parallel(l, n, delta_freq_quadrat, magnetic_field_s, magnetic_field_sprime)
    name_string=f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)
    np.savetxt(DATA_DIR, sme, delimiter=' ')
    print(f'Supermatrix saved to {DATA_DIR}')
    #Calculate and save frequency shifts for (l,n)-modes
    f_shifts=fs.calculate_safe_extract_freq_shifts(l,n)
    print(f'Results of Frequency shifts for l={l} n={n} multiplet:')
    print(f_shifts)
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")
    print('Finished.\n\n\n')

def calculate_freq_shift_quasi_degenerate_first_approx(l,n,delta_freq_quadrat,magnetic_field_s, magnetic_field_sprime=None):
    if magnetic_field_sprime == None:
        magnetic_field_sprime = magnetic_field_s

    start_time = time.time()
    #Create and Save index map
    index_map = GMEP.create_index_map(l,n,GMEP.eigenspace(l,n,delta_freq_quadrat, eigen_tag='FirstApprox'))
    fs_1a.save_index_map_to_file(index_map, l,n)

    #Calculate and save supermatrix
    #ALL GME involved are safed as well for later use
    sme = GMEP.supermatrix_parallel(l, n, delta_freq_quadrat, magnetic_field_s, magnetic_field_sprime, eigen_tag='FirstApprox')
    name_string=f'supermatrix_array_{l}_{n}_first_approx.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)
    np.savetxt(DATA_DIR, sme, delimiter=' ')
    print(f'Supermatrix approximation saved to {DATA_DIR}')
    #Calculate and save frequency shifts for (l,n)-modes
    f_shifts=fs_1a.calculate_safe_extract_freq_shifts(l,n)
    print(f'Results of Frequency shifts for l={l} n={n} multiplet in first approximation:')
    print(f_shifts)
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")
    print('Finished.\n\n\n')


def calculate_freq_shift_quasi_degenerate_second_approx(l,n,delta_freq_quadrat,magnetic_field_s, magnetic_field_sprime=None):
    if magnetic_field_sprime == None:
        magnetic_field_sprime = magnetic_field_s

    start_time = time.time()
    #Create and Save index map
    index_map = GMEP.create_index_map(l,n,GMEP.eigenspace(l,n,delta_freq_quadrat, eigen_tag='SelfCoupling'))
    fs_2a.save_index_map_to_file(index_map, l,n)

    #Calculate and save supermatrix
    #ALL GME involved are safed as well for later use
    sme = GMEP.supermatrix_parallel(l, n, delta_freq_quadrat, magnetic_field_s, magnetic_field_sprime, eigen_tag='SelfCoupling')
    name_string=f'supermatrix_array_{l}_{n}_second_approx.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)
    np.savetxt(DATA_DIR, sme, delimiter=' ')
    print(f'Supermatrix approximation saved to {DATA_DIR}')
    #Calculate and save frequency shifts for (l,n)-modes
    f_shifts=fs_2a.calculate_safe_extract_freq_shifts(l,n)
    print(f'Results of Frequency shifts for l={l} n={n} multiplet in second approximation:')
    print(f_shifts)
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")
    print('Finished.\n\n\n')

def main():
    #initialize magnetic field
    #The programm is designed to handle only one Model of the magnetic fields at once.
    #If the Model is changed, clean the Output folders and re-run.
    magnetic_field_modelA = radial_kernels.MagneticField(B_max=300, mu=0.713, sigma=0.05, s=2)

    #delta_freq^2 = 700 microHz^2
    delta_freq_quadrat = 700  #microHz^2


    #Over all l,n in parameterspace of FirstApprox (excludes multiplets which do not penetrate into effective range of magnetic field
    for l in range(2,140+1):
        for n in range(0, 35+1):
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
                print(f'Computing multiplet l={l}, n={n}, freq={GMEP.frequencies_GYRE(l,n)} microHz')
                #Calculate using First approximation
                calculate_freq_shift_quasi_degenerate_first_approx(l,n,delta_freq_quadrat, magnetic_field_modelA)
    print(f'All multiplets finished.')


    '''
    #For master thesis
    l = 5
    n = [6,12,18]

    for value in n:
        print(f'Computing l={l}, n={value}, freq={GMEP.frequencies_GYRE(l,value)} microHz')
        calculate_freq_shift_quasi_degenerate_second_approx(l,value,delta_freq_quadrat, magnetic_field_modelA)
    '''

    '''
    #Potential follow up multiplets
    l = 25
    n = [0,6,13]
    for value in n:
        print(f'Computing l={l}, n={value}, freq={GMEP.frequencies_GYRE(l,value)} microHz')
        calculate_freq_shift_quasi_degenerate(l,value,delta_freq_quadrat, magnetic_field_modelA)

    l = 50
    n = [0,5,9]
    for value in n:
        print(f'Computing l={l}, n={value}, freq={GMEP.frequencies_GYRE(l,value)} microHz')
        calculate_freq_shift_quasi_degenerate(l,value,delta_freq_quadrat, magnetic_field_modelA)

    l = 100
    n = [0,3,6]
    for value in n:
        print(f'Computing l={l}, n={value}, freq={GMEP.frequencies_GYRE(l,value)} microHz')
        calculate_freq_shift_quasi_degenerate(l,value,delta_freq_quadrat, magnetic_field_modelA)

    l = 150
    n = [0,4]
    for value in n:
        print(f'Computing l={l}, n={value}, freq={GMEP.frequencies_GYRE(l,value)} microHz')
        calculate_freq_shift_quasi_degenerate(l,value,delta_freq_quadrat, magnetic_field_modelA)
    '''

if __name__== '__main__':
    main()
