from sympy.physics.wigner import wigner_3j as wig
import numpy as np
import math
from constants import gamma as ga
from scipy.special import gamma

import time
from Integrals_over_gsh_old import integral_over_3gsh as int_3gsh_old
from Integrals_over_gsh_old import integral_over_4gsh as int_4gsh_old
from Integrals_over_gsh_old import integral_over_5gsh as int_5gsh_old
from Integrals_over_gsh_old import integral_over_6gsh as int_6gsh_old

def integral_associated_LP(l,m):
    #Calculates integral over associated Legendre Polynom from -1 to 1
    if abs(m) > l or (l+m) %2 == 1:
        return 0.0
    elif l == 0:
        return 2
    elif m >= 0:
        return ((-1)**m+(-1)**l)*2**(m-2)*m*gamma(l/2)*gamma((l+m+1)/2)/(math.factorial(int((l-m)/2))*gamma((l+3)/2))
    elif m < 0:
        m = abs(m)
        return (-1)**m*math.factorial(l-m)/math.factorial(l+m)*((-1)**m+(-1)**l)*2**(m-2)*m*gamma(l/2)*gamma((l+m+1)/2)/(math.factorial(int((l-m)/2))*gamma((l+3)/2))
    else:
        return "Selected l is not an positive integer"

def integral_GLP(l,N):
    #Calculates integral over genralised Legendre Polynom from -1 to 1 and with m=0
    integral = (-1)**N*np.sqrt(math.factorial(l-N)/math.factorial(l+N))*integral_associated_LP(l,N)
    return integral


def integral_over_3gsh(l, N, m):
    # Delta_function m1=m2+m3
    if m[0] != m[1] + m[2]:
        return 0.0
    # Property of wigner3j |m|,|N|<=l
    if abs(m[0]) > l[0] or abs(m[1]) > l[1] or abs(m[2]) > l[2] or abs(N[0]) > l[0] or abs(N[1]) > l[1] or abs(N[2]) > l[2]:
        return 0.0

    # Dahlen & Tromp 1998 formula for the special case of N1=N2+N3
    if N[0] == N[1]+N[2]:
        return float(4*np.pi*(-1)**(N[0]+m[0])*ga(l[0])*ga(l[1])*ga(l[2])*wig(l[0], l[1], l[2], -N[0], N[1], N[2])*wig(l[0], l[1], l[2], -m[0], m[1], m[2]))

    # Property of wigner3j; triangle condition
    minl = abs(l[1]-l[2])
    maxl = l[1]+l[2]
    larray = list(range(minl, maxl + 1))

    result = 0.0

    for ls in larray:
        # Property of wigner3j |m|,|N|<=l
        if abs(-N[1] - N[2]) > ls or abs(-m[0]) > ls:
            continue
        # Property of wigner3j l1+l2+l3 must be even if m1=m2=m3=0
        if all(val == 0 for val in m) and (ls + l[1] + l[2]) % 2 == 1:
            continue
        if all(val == 0 for val in N[1:]) and (ls + l[1] + l[2]) % 2 == 1:
            continue

        # Property of wigner3j; triangle condition
        minlp = abs(l[0] - ls)
        maxlp = l[0] + ls
        lparray = list(range(minlp, maxlp + 1))

        for lp in lparray:
            # Property of wigner3j |m|,|N|<=l
            if abs(-N[0]+N[1]+N[2]) > lp:
                continue
            # Property of wigner3j l1+l2+l3 must be even if m1=m2=m3=0
            if m[0] == 0 and (lp+l[0]+ls) %2 == 1:
                continue
            if N[0] == -N[1]-N[2] == 0 and (lp+l[0]+ls) %2 == 1:
                continue

            #Shouldn't get zero, since all wigner3j selection rules are already satisfied
            wigner1 = wig(ls, l[1], l[2], -N[1]-N[2], N[1], N[2])
            wigner2 = wig(ls, l[1], l[2], -m[1]-m[2], m[1], m[2])
            wigner3 = wig(lp, l[0], ls, -N[0]+N[1]+N[2], N[0], -N[1]-N[2])
            wigner4 = wig(lp, l[0], ls, 0, m[0], -m[0])

            term = ga(ls)**2*ga(lp)**2*wigner1*wigner2*wigner3*wigner4*integral_GLP(lp,-N[0]+N[1]+N[2])
            result += term

    #Common factor for all terms
    return 2*np.pi*(4*np.pi)**2*ga(l[0])*ga(l[1])*ga(l[2])*result

def integral_over_4gsh(l, N, m):

    # Property of wigner3j |m|,|N|<=l
    if abs(N[3]) > l[3] or abs(m[3]) > l[3]:
        return 0.0

    # Property of wigner3j; triangle condition
    minl = abs(l[2]-l[3])
    maxl = l[2]+l[3]
    larray = list(range(minl, maxl + 1))

    result = 0.0

    for ls in larray:
        # Property of wigner3j |m|,|N|<=l
        if abs(-N[2]-N[3]) > ls or abs(-m[2]-m[3]) > ls:
            continue
        # Property of wigner3j l1+l2+l3 must be even if m1=m2=m3=0
        if all(val == 0 for val in m[2:]) and (ls + l[2] + l[3]) % 2 == 1:
            continue
        if all(val == 0 for val in N[2:]) and (ls + l[2] + l[3]) % 2 == 1:
            continue

        # Shouldn't get zero, since all wigner3j selection rules are already satisfied
        wigner1 = wig(ls, l[2], l[3], -N[2]-N[3], N[2], N[3])
        wigner2 = wig(ls, l[2], l[3], -m[2]-m[3], m[2], m[3])

        l_list = [l[0], l[1], ls]
        m_list = [m[0], m[1], m[2]+m[3]]
        N_list = [N[0], N[1], N[2]+N[3]]
        term = ga(ls)*wigner1*wigner2*integral_over_3gsh(l_list, N_list, m_list)
        result += term

    # Common factor for all terms
    return 4*np.pi*(-1)**(N[2]+N[3]+m[2]+m[3])*ga(l[2])*ga(l[3])*result


def integral_over_5gsh(l, N, m):

    # Property of wigner3j |m|,|N|<=l
    if abs(N[4]) > l[4] or abs(m[4]) > l[4]:
        return 0.0

    minl = abs(l[3]-l[4])
    maxl = l[3]+l[4]
    larray = list(range(minl, maxl + 1))

    result = 0.0

    for ls in larray:

        # Property of wigner3j |m|,|N|<=l
        if abs(-N[3]-N[4]) > ls or abs(-m[3]-m[4]) > ls:
            continue
        # Property of wigner3j l1+l2+l3 must be even if m1=m2=m3=0
        if all(val == 0 for val in m[3:]) and (ls + l[3] + l[4]) % 2 == 1:
            continue
        if all(val == 0 for val in N[3:]) and (ls + l[3] + l[4]) % 2 == 1:
            continue

        # Shouldn't get zero, since all wigner3j selection rules are already satisfied
        wigner1 = wig(ls, l[3], l[4], -N[3]-N[4], N[3], N[4])
        wigner2 = wig(ls, l[3], l[4], -m[3]-m[4], m[3], m[4])

        l_list = [l[0], l[1], l[2], ls]
        m_list = [m[0], m[1], m[2], m[3]+m[4]]
        N_list = [N[0], N[1], N[2], N[3]+N[4]]
        term = ga(ls)*wigner1*wigner2*integral_over_4gsh(l_list, N_list, m_list)
        result += term

    # Common factor for all terms
    return 4*np.pi*(-1)**(N[3]+N[4]+m[3]+m[4])*ga(l[3])*ga(l[4])*result


def integral_over_6gsh(l, N, m):

    # Property of wigner3j |m|,|N|<=l
    if abs(N[5]) > l[5] or abs(m[5]) > l[5]:
        return 0.0

    minl = abs(l[4]-l[5])
    maxl = l[4]+l[5]
    larray = list(range(minl, maxl + 1))

    result = 0.0

    for ls in larray:

        # Property of wigner3j |m|,|N|<=l
        if abs(-N[4]-N[5]) > ls or abs(-m[4]-m[5]) > ls:
            continue
        # Property of wigner3j l1+l2+l3 must be even if m1=m2=m3=0
        if all(val == 0 for val in m[4:]) and (ls + l[4] + l[5]) % 2 == 1:
            continue
        if all(val == 0 for val in N[4:]) and (ls + l[4] + l[5]) % 2 == 1:
            continue

        # Shouldn't get zero, since all wigner3j selection rules are already satisfied
        wigner1 = wig(ls, l[4], l[5], -N[4]-N[5], N[4], N[5])
        wigner2 = wig(ls, l[4], l[5], -m[4]-m[5], m[4], m[5])

        l_list = [l[0], l[1], l[2], l[3], ls]
        m_list = [m[0], m[1], m[2], m[3], m[4]+m[5]]
        N_list = [N[0], N[1], N[2], N[3], N[4]+N[5]]
        term = ga(ls)*wigner1*wigner2*integral_over_5gsh(l_list, N_list, m_list)
        result += term

    return 4*np.pi*(-1)**(N[4]+N[5]+m[4]+m[5])*ga(l[4])*ga(l[5])*result

def main():
    pass
    # TEST AREA
    '''
    #3 GSH
    l1=50
    l2=4
    l3=2
    N1=3
    N2=3
    N3=1
    m1=-1
    m2=0
    m3=-1
    start_time = time.time()
    int_new=float(integral_over_3gsh([l1,l2,l3],[N1,N2,N3],[m1,m2,m3]))
    print(int_new)
    end_time = time.time()
    print("Elapsed time new formula:", end_time - start_time, "seconds")
    
    start_time = time.time()
    int_old=float(int_3gsh_old([l1,l2,l3],[N1,N2,N3],[m1,m2,m3]))
    print(int_old)
    end_time = time.time()
    print("Elapsed time old formula:", end_time - start_time, "seconds")
    diff=abs(int_new-int_old)
    print('Differenze = ', float(diff))
    '''

    '''
    #4 GSH
    l1=5
    l2=4
    l3=4
    l4=2
    N1=4
    N2=-4
    N3=0
    N4=-2
    m1=-2
    m2=-1
    m3=0
    m4=-1
    start_time = time.time()
    int_new=float(integral_over_4gsh([l1,l2,l3,l4],[N1,N2,N3,N4],[m1,m2,m3,m4]))
    print(int_new)
    end_time = time.time()
    print("Elapsed time new formula:", end_time - start_time, "seconds")
    
    start_time = time.time()
    int_old=float(int_4gsh_old([l1,l2,l3,l4],[N1,N2,N3,N4],[m1,m2,m3,m4]))
    print(int_old)
    end_time = time.time()
    print("Elapsed time old formula:", end_time - start_time, "seconds")
    diff=abs(int_new-int_old)
    print('Differenze = ', float(diff))
    '''
    '''
    #5 GSH
    l1=5
    l2=3
    l3=4
    l4=1
    l5=2
    N1=3
    N2=-2
    N3=0
    N4=-1
    N5=2
    m1=3
    m2=1
    m3=0
    m4=1
    m5=1
    start_time = time.time()
    int_new=float(integral_over_5gsh([l1,l2,l3,l4,l5],[N1,N2,N3,N4,N5],[m1,m2,m3,m4,m5]))
    print(int_new)
    end_time = time.time()
    print("Elapsed time new formula:", end_time - start_time, "seconds")
    
    start_time = time.time()
    int_old=float(int_5gsh_old([l1,l2,l3,l4,l5],[N1,N2,N3,N4,N5],[m1,m2,m3,m4,m5]))
    print(int_old)
    end_time = time.time()
    print("Elapsed time old formula:", end_time - start_time, "seconds")
    diff=abs(int_new-int_old)
    print('Differenze = ', float(diff))
    '''


    '''
    #6 GSH
    l1=5
    l2=3
    l3=4
    l4=1
    l5=2
    l6=1
    N1=3
    N2=-2
    N3=0
    N4=-1
    N5=2
    N6=1
    m1=3
    m2=1
    m3=0
    m4=1
    m5=1
    m6=0
    start_time = time.time()
    int_new=float(integral_over_6gsh([l1,l2,l3,l4,l5,l6],[N1,N2,N3,N4,N5,N6],[m1,m2,m3,m4,m5,m6]))
    print(int_new)
    end_time = time.time()
    print("Elapsed time new formula:", end_time - start_time, "seconds")
    
    start_time = time.time()
    int_old=float(int_6gsh_old([l1,l2,l3,l4,l5,l6],[N1,N2,N3,N4,N5,N6],[m1,m2,m3,m4,m5,m6]))
    print(int_old)
    end_time = time.time()
    print("Elapsed time old formula:", end_time - start_time, "seconds")
    diff=abs(int_new-int_old)
    print('Differenze = ', float(diff))
    '''

if __name__== '__main__':
    main()