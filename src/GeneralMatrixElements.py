import numpy as np
import angular_kernels
import radial_kernels
import os
import pygyre as pg

def single_GM(l,n,m,lprime,nprime,mprime,magnetic_field_s, magnetic_field_sprime=None):
    if magnetic_field_sprime == None:
        magnetic_field_sprime = magnetic_field_s
    if m != mprime:
        return 0.0

    s = magnetic_field_s.s
    sprime = magnetic_field_sprime.s
    radius_array = radial_kernels.MESA_structural_data()[0]
    R_args = (l, n, lprime, nprime, radius_array, magnetic_field_s, magnetic_field_sprime)
    S_args = (lprime, l, s, sprime, mprime, m)
    
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

    return general_matrix_element


def normalization(l,n):
    radius_array = radial_kernels.MESA_structural_data()[0]
    R_sun = radial_kernels.MESA_structural_data()[3]  # R_sun in cm
    rho_0=radial_kernels.MESA_structural_data()[2]*R_sun**3 #g/R_sun^3
    func=rho_0*(radial_kernels.eigenfunctions(l,n,radius_array)[0]**2+l*(l+1)*radial_kernels.eigenfunctions(l,n,radius_array)[1]**2)*radius_array**2
    return radial_kernels.radial_integration(radius_array, func)  #g*R_sun^2

def frequencies_GYRE(l,n):
    name_string="summary_solar_test_suite.h5"
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE')
    summary_file = pg.read_output(os.path.join(DATA_DIR, name_string))
    l_group = summary_file.group_by('l')
    filterd_freq=next(value for value in l_group.groups[l] if value['n_pg']==n)['freq'].real

    return filterd_freq  #microHz

def eigenspace(l,n, delta_omega):
    #delta_omega in microHz
    omega_ref=frequencies_GYRE(l,n)
    name_string="summary_solar_test_suite.h5"
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE')
    summary_file = pg.read_output(os.path.join(DATA_DIR, name_string))
    K_space=[]
    for row in summary_file:
        if abs(row['freq'].real-omega_ref) <= delta_omega:
            new_row = {
                'freq': row['freq'],
                'n': row['n_pg'],
                'l': row['l']}
            K_space.append(new_row)
    '''
    for i in range(0,len(K_space)):
        print(K_space[i]['freq'].real, K_space[i]['l'], K_space[i]['n'])
    '''
    return K_space

def supermatrix_element(omega_ref,l,n,m,lprime,nprime,mprime, magnetic_field_s, magnetic_field_sprime=None):
    if magnetic_field_sprime == None:
        magnetic_field_sprime = magnetic_field_s

    gme = single_GM(l,n,m,lprime,nprime,mprime,magnetic_field_s,magnetic_field_sprime)
    normal=normalization(l,n)
    if l==lprime and n==nprime and m==mprime:
        delta=1
    else:
        delta=0
    sme=gme/normal*10**12*10**6*radial_kernels.MESA_structural_data()[3]-(omega_ref**2-frequencies_GYRE(l,n)**2)*delta

    return sme, gme, normal

def supermatrix(l,n,delta_omega,magnetic_field_s, magnetic_field_sprime=None):
    if magnetic_field_sprime == None:
        magnetic_field_sprime = magnetic_field_s

    omega_ref=frequencies_GYRE(l,n)
    K_space=eigenspace(l,n, delta_omega)

    size=0
    for i in range(0,len(K_space)):
        for iprime in range(0, len(K_space)):
            size=size+(2*K_space[i]['l']+1)*(2*K_space[iprime]['l']+1)
    if np.sqrt(size) %1 ==0:
        matrix_size=int(np.sqrt(size))
    else:
        return ValueError('Matrixsize not an integer')
    #print(size,matrix_size)
    supermatrix_array = np.empty((matrix_size,matrix_size), dtype=np.float64)

    # fill supermatrix
    row = 0
    for k in range(0, len(K_space)):
        l = K_space[k]['l']
        n = K_space[k]['n']
        for m in range(0, 2*l+1):
            m_index = m
            m = m_index-l
            col = 0
            for kprime in range(0, len(K_space)):
                lprime = K_space[kprime]['l']
                nprime = K_space[kprime]['n']
                for mprime in range(0,2*lprime+1):
                    mprime_index = mprime
                    mprime = mprime_index-lprime
                    # supermatrix_array[row,col] = (l,n,m,lprime,nprime,mprime) # change dtype=object for testing
                    sme = supermatrix_element(omega_ref,l,n,m,lprime,nprime,mprime,magnetic_field_s)[0]
                    supermatrix_array[row,col] = sme
                    col += 1
            row += 1
    return supermatrix_array


def main():
    #initialize magnetic field
    magnetic_field_modelA = radial_kernels.MagneticField(B_max=50, mu=0.7, sigma=0.04, s=2)

    #Delta Omega^2 = 30 microHz^2
    delta_omega = np.sqrt(30)  #microHz

    #Omega_ref
    l=5
    n=5
    #freq=frequencies_GYRE(l,n)

    #Eigenspace
    K_space=eigenspace(l,n,delta_omega)

    #Supermatrix
    supermatrix_array=supermatrix(5,5, delta_omega, magnetic_field_modelA)
    name_string=f'supermatrix_array_serial_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Supermatrices', name_string)
    np.savetxt(DATA_DIR, supermatrix_array, delimiter=' ')

    '''
    #TEST OF SINGLE FUNCTIONS
    l=20
    n=5
    m=1
    lprime=20
    nprime=5
    mprime=1
    freq=frequencies_GYRE(l,n)  #microHz
    print('freq =', freq, 'microHz')
    print('freq^2 =', freq**2, 'microHz^2')
    gme=single_GM(l,n,m,lprime,nprime,mprime,magnetic_field_modelA)
    normal=normalization(l,n)
    print('Normalisation =', normal, 'g*R_sun^2')
    print(f'H_{l} {n} {m},{lprime} {nprime} {mprime} =', gme, ' kG^2*R_Sun^3')
    print(f'H_{l} {n} {m},{lprime} {nprime} {mprime}/ normal =', gme/normal*10**12*10**6*radial_kernels.MESA_structural_data()[3], ' microHz^2')
    print(np.sqrt(freq**2+gme/normal*10**12*10**6*radial_kernels.MESA_structural_data()[3]), 'microHz')
    #solution for l=lprime=2, n=5, nprime=4, m=mprime=1:  H_k,k'=-2259.118191997141 kG^2*R_Sun^3
    approx_frequency_shift=(np.sqrt(freq**2+gme/normal*10**12*10**6*radial_kernels.MESA_structural_data()[3])-freq)/(2*np.pi)  #microHz, after Kiefer2018
    print(f'approx frequency shift =', approx_frequency_shift, 'microHz')
    print(f'approx frequency shift =', approx_frequency_shift*10**3, 'nHz')
    #sme,gme,normal=supermatrix_element(freq, 5, 5, 1, 3, 16, 1, magnetic_field_modelA)
    #print(sme,gme,normal)
    '''
if __name__== '__main__':
    main()
