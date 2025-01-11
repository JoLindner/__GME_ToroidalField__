from sympy.physics.wigner import wigner_3j as wig
import numpy as np
from constants import gamma as ga
from scipy.special import gammaln


def integral_associated_LP(l,m):
    #Calculates integral over associated Legendre Polynom from -1 to 1
    if abs(m) > l or (l+m) %2 == 1:
        return 0.0
    elif l == 0:
        return 2.0, 1
    elif m >= 0:
        if m ==0:
            return 0.0
        else:
         integral_abs = np.log(abs((-1.0)**m+(-1.0)**l))+(m-2)*np.log(2)+np.log(m)+gammaln(l/2)+gammaln((l+m+1)/2)-gammaln((l-m)/2+1)-gammaln((l+3)/2)
         if m%2 == 1 and l%2 == 1:
             sign = -1
         else:
             sign = 1
        return integral_abs, sign
    elif m < 0:
        m = abs(m)
        integral_abs= gammaln(l-m+1)-gammaln(l+m+1)+np.log(abs((-1.0)**m+(-1.0)**l))+(m-2)*np.log(2)+np.log(m)+gammaln(l/2)+gammaln((l+m+1)/2)-gammaln((l-m)/2+1)-gammaln((l+3)/2)
        sign = 1
        return integral_abs, sign
    else:
        return "Selected l is not an positive integer"


def integral_GLP(l,N):
    #Calculates integral over genralised Legendre Polynom from -1 to 1 and with m=0
    #calculate logs to avoid overflows
    if integral_associated_LP(l,N) == 0:
        return 0.0
    integral_abs = 1.0/2.0*(gammaln(l-N+1)-gammaln(l+N+1))+integral_associated_LP(l,N)[0]
    sign = (-1)**N*integral_associated_LP(l,N)[1]
    integral = sign*np.exp(integral_abs)
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
        return float(4*np.pi*(-1.0)**(N[0]+m[0])*ga(l[0])*ga(l[1])*ga(l[2])*wig(l[0], l[1], l[2], -N[0], N[1], N[2])*wig(l[0], l[1], l[2], -m[0], m[1], m[2]))

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
    return 4*np.pi*(-1.0)**(N[2]+N[3]+m[2]+m[3])*ga(l[2])*ga(l[3])*result


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
    return 4*np.pi*(-1.0)**(N[3]+N[4]+m[3]+m[4])*ga(l[3])*ga(l[4])*result


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

    return 4*np.pi*(-1.0)**(N[4]+N[5]+m[4]+m[5])*ga(l[4])*ga(l[5])*result


def main():
    pass


if __name__ == '__main__':
    main()