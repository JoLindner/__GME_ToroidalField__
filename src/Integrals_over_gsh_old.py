from sympy.physics.wigner import wigner_3j as wig
import numpy as np
from constants import gamma as ga
from GLP.GLP import generalized_legendre_polynom as glp
import time


def integral_glp(l, N, m):
    resolution = 4000
    theta_array = np.linspace(0, np.pi, resolution+1, endpoint=False)
    theta_array = theta_array[1:]
    
    #selection rules for odd GLP
    if m==0 and (l+N) %2 ==1:
        return 0.0
    if N==0 and (l+m) %2 ==1:
        return 0.0
    
    func = glp(l, N, m, resolution)*np.sin(theta_array)
    integral = 0.0
    

    for i in range(1, resolution):
        summand = (func[i-1]+func[i])/2*(theta_array[i]-theta_array[i-1])
        integral += summand

    return float(integral)

def integral_over_3gsh(l, N, m):

    if m[0] != m[1]+m[2]:
        print('This integral should be 0!')
        return 0.0

    if m[0] > l[0] or m[1] > l[1] or m[2] > l[2] or N[0] > l[0] or N[1] > l[1] or N[2] > l[2]:
        print('This integral should be 0!')
        return 0.0

    # Dahlen & Tromp 1998
    if N[0] == N[1]+N[2]:
        return float(4*np.pi*(-1)**(N[0]+m[0])*ga(l[0])*ga(l[1])*ga(l[2])*wig(l[0], l[1], l[2], -N[0], N[1], N[2])*wig(l[0], l[1], l[2], -m[0], m[1], m[2]))

    minl = abs(l[1]-l[2])
    maxl = l[1]+l[2]
    lengthl = abs(maxl-minl)
    larray = [minl+i for i in range(lengthl+1)]

    result = 0.0

    for i in range(len(larray)):
        ls = larray[i]  # l

        if abs(-N[1]-N[2]) > ls or m[0] > ls:
            #print('This term violates selection rule 1! ls=',ls)
            continue
        if m[0]==m[1]==m[2]==0 and (ls+l[1]+l[2]) %2 == 1:
            #print('This term violates selection rule 2! ls=',ls)
            continue
            
        minlp = abs(l[0]-ls)
        maxlp = l[0]+ls
        lengthlp = abs(maxlp-minlp)
        lparray = [minlp+i for i in range(lengthlp+1)]

        for ip in range(len(lparray)):
            lp = lparray[ip]  # l'

            if abs(-N[0]-N[1]-N[2]) > lp or abs(-2*m[0]) > lp:
                #print('This term violates selection rule 3! ls=',ls,'lp=',lp)
                continue
            if m[0]==m[1]==m[2]==0 and (lp+l[0]+ls) %2 == 1:
                #print('This term violates selection rule 4! ls=',ls,'lp=',lp)
                continue


            wigner1 = wig(ls, l[1], l[2], -N[1]-N[2], N[1], N[2])
            if wigner1 == 0:
                continue
            wigner2 = wig(ls, l[1], l[2], -m[1]-m[2], m[1], m[2])
            if wigner2 == 0:
                continue
            wigner3 = wig(lp, l[0], ls, -N[0]-N[1]-N[2], N[0], N[1]+N[2])
            if wigner3 == 0:
                continue
            wigner4 = wig(lp, l[0], ls, -2*m[0], m[0], m[0])
            if wigner4 == 0:
                continue

            term = 2*np.pi*(4*np.pi)**2*(-1)**(N[1]+N[2]+m[1]+m[2])*ga(ls)**2*ga(lp)**2*ga(l[0])*ga(
                l[1])*ga(l[2])*wigner1*wigner2*wigner3*wigner4*integral_glp(lp, -N[0]-N[1]-N[2], -2*m[0])

            result += term
            #print(float(term), 'ls=',ls, 'lp=',lp)

    return float(result)


def integral_over_4gsh(l, N, m):
    minl = abs(l[2]-l[3])
    maxl = l[2]+l[3]
    lengthl = abs(maxl-minl)
    larray = [minl+i for i in range(lengthl+1)]

    result = 0.0

    for i in range(len(larray)):
        ls = larray[i]  # l

        wigner1 = wig(ls, l[2], l[3], -N[2]-N[3], N[2], N[3])
        if wigner1 == 0:
            continue
        wigner2 = wig(ls, l[2], l[3], -m[2]-m[3], m[2], m[3])
        if wigner2 == 0:
            continue

        l_list = [l[0], l[1], ls]
        m_list = [m[0], m[1], m[2]+m[3]]
        N_list = [N[0], N[1], N[2]+N[3]]
        term = 4*np.pi*(-1)**(N[2]+N[3]+m[2]+m[3])*ga(ls)*ga(l[2])*ga(l[3]) * \
            wigner1*wigner2*integral_over_3gsh(l_list, N_list, m_list)
        result += term

    return float(result)


def integral_over_5gsh(l, N, m):
    minl = abs(l[3]-l[4])
    maxl = l[3]+l[4]
    lengthl = abs(maxl-minl)
    larray = [minl+i for i in range(lengthl+1)]

    result = 0.0

    for i in range(len(larray)):
        ls = larray[i]  # l

        wigner1 = wig(ls, l[3], l[4], -N[3]-N[4], N[3], N[4])
        if wigner1 == 0:
            continue
        wigner2 = wig(ls, l[3], l[4], -m[3]-m[4], m[3], m[4])
        if wigner2 == 0:
            continue

        l_list = [l[0], l[1], l[2], ls]
        m_list = [m[0], m[1], m[2], m[3]+m[4]]
        N_list = [N[0], N[1], N[2], N[3]+N[4]]
        term = 4*np.pi*(-1)**(N[3]+N[4]+m[3]+m[4])*ga(ls)*ga(l[3])*ga(l[4]) * \
            wigner1*wigner2*integral_over_4gsh(l_list, N_list, m_list)
        result += term

    return float(result)


def integral_over_6gsh(l, N, m):
    minl = abs(l[4]-l[5])
    maxl = l[4]+l[5]
    lengthl = abs(maxl-minl)
    larray = [minl+i for i in range(lengthl+1)]

    result = 0.0

    for i in range(len(larray)):
        ls = larray[i]  # l

        wigner1 = wig(ls, l[4], l[5], -N[4]-N[5], N[4], N[5])
        if wigner1 == 0:
            continue
        wigner2 = wig(ls, l[4], l[5], -m[4]-m[5], m[4], m[5])
        if wigner2 == 0:
            continue

        l_list = [l[0], l[1], l[2], l[3], ls]
        m_list = [m[0], m[1], m[2], m[3], m[4]+m[5]]
        N_list = [N[0], N[1], N[2], N[3], N[4]+N[5]]
        term = 4*np.pi*(-1)**(N[4]+N[5]+m[4]+m[5])*ga(ls)*ga(l[4])*ga(l[5]) * \
            wigner1*wigner2*integral_over_5gsh(l_list, N_list, m_list)
        result += term

    return float(result)

def main():
    pass
    '''
    # TEST_AREA
    l1 = 4
    l2 = 4
    l3 = 4
    l4 = 4
    l = [l1, l2, l3, l4]
    m1 = 2
    m2 = 1
    m3 = 1
    m4 = 0
    m = [m1, m2, m3, m4]
    N1 = 2
    N2 = 1
    N3 = 1
    N4 = -1
    N = [N1, N2, N3, N4]
    
    start_time = time.time()
    test1 = integral_over_4gsh(l, N, m)
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")
    print(test1)
    '''
    '''
    #print(integral_glp(1,1,0))
    
    l_list=[1,1,1,1]
    m_list=[0,0,0,0]
    N_list=[0,0,1,1]
    
    print(integral_over_4gsh(l_list, N_list, m_list))
    
    
    
    l_list=[1,1,1,1,1]
    m_list=[1,0,1,0,0]
    N_list=[0,0,1,1,1]
    
    print(integral_over_5gsh(l_list, N_list, m_list))
    
    
    l_list=[1,1,1,1,1,1]
    m_list=[1,1,0,0,0,0]
    N_list=[0,0,1,1,1,1]
    
    print(integral_over_6gsh(l_list, N_list, m_list))
    '''

if __name__== '__main__':
    main()