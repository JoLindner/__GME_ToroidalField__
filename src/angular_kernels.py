from Integral_over_gsh import integral_over_4gsh as integral4
from Integral_over_gsh import integral_over_5gsh as integral5
from Integral_over_gsh import integral_over_6gsh as integral6
import numpy as np
from constants import c_const as co
import time


def S1(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]
    s1=co(s,0)*co(sprime,0)*integral4(l_list,[0,0,1,1],m_list)
        
    return float(s1)


def S2(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s2=1.0/4*co(s,0)*co(sprime,0)*co(l,0)*(co(s,0)*integral4(l_list, [0,-1,0,1], m_list)\
                                                        -co(s,0)*integral4(l_list, [0,1,0,1], m_list)\
                                                        +co(s,2)*integral4(l_list, [0,-1,2,1], m_list)\
                                                        -co(s,2)*integral4(l_list, [0,1,2,1], m_list))
    return float(s2)


def S3(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s3=1.0/4*co(l,0)*co(s,0)*co(sprime,0)*(co(l,2)*integral4(l_list, [0,2,1,1], m_list)\
                                           +co(l,2)*integral4(l_list, [0,-2,1,1], m_list)\
                                           -2*co(l,0)*integral4(l_list, [0,0,1,1], m_list))
    return float(s3)

def S4(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s4=-1.0/4*co(l,0)*co(lprime,0)*co(s,0)*co(sprime,0)*(integral4(l_list, [-1,-1,1,1], m_list)\
                                                         +integral4(l_list, [-1,1,1,1], m_list)\
                                                         +integral4(l_list, [1,-1,1,1], m_list)\
                                                         +integral4(l_list, [1,1,1,1], m_list))
    return float(s4)

def S5(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s5=1.0/4*co(l,0)*co(s,0)*co(sprime,0)*(co(sprime,0)*integral4(l_list, [0,-1,1,0], m_list)\
                                           -co(sprime,0)*integral4(l_list, [0,1,1,0], m_list)\
                                           -co(sprime,2)*integral4(l_list, [0,-1,1,2], m_list)\
                                           +co(sprime,2)*integral4(l_list, [0,1,1,2], m_list))
    return float(s5)

def S6(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s6=1.0/4*co(l,0)*co(s,0)*co(sprime,0)*(co(s,0)*integral4(l_list, [0,-1,0,1], m_list)\
                                           -co(s,0)*integral4(l_list, [0,1,0,1], m_list)\
                                           -co(s,2)*integral4(l_list, [0,-1,2,1], m_list)\
                                           +co(s,2)*integral4(l_list, [0,1,2,1], m_list))
    return float(s6)


def S7(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s7=1.0/4*co(lprime,0)*co(s,0)*co(sprime,0)*(co(s,0)*integral4(l_list, [-1,0,0,1], m_list)\
                                                        +co(s,2)*integral4(l_list, [-1,0,2,1], m_list)\
                                                        -co(s,0)*integral4(l_list, [1,0,0,1], m_list)\
                                                        -co(s,2)*integral4(l_list, [1,0,2,1], m_list))
    return float(s7)



def S8(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s8=1.0/4*co(lprime,0)*co(s,0)*co(sprime,0)*(integral4(l_list, [-1,0,0,1], m_list)\
                                                -integral4(l_list, [-1,0,2,1], m_list)\
                                                -integral4(l_list, [1,0,0,1], m_list)\
                                                +integral4(l_list, [1,0,2,1], m_list))
    return float(s8)


def S9(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s9=1.0/16*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(co(s,0)*co(sprime,0)*integral4(l_list, [-1,-1,0,0], m_list)\
                                                        +co(s,0)*co(sprime,2)*integral4(l_list, [-1,-1,0,2], m_list)\
                                                        +co(s,2)*co(sprime,0)*integral4(l_list, [-1,-1,2,0], m_list)\
                                                        +co(s,2)*co(sprime,2)*integral4(l_list, [-1,-1,2,2], m_list)\
                                                        -co(s,0)*co(sprime,0)*integral4(l_list, [-1,1,0,0], m_list)\
                                                        -co(s,0)*co(sprime,2)*integral4(l_list, [-1,1,0,2], m_list)\
                                                        -co(s,2)*co(sprime,0)*integral4(l_list, [-1,1,2,0], m_list)\
                                                        -co(s,2)*co(sprime,2)*integral4(l_list, [-1,1,2,2], m_list)\
                                                        -co(s,0)*co(sprime,0)*integral4(l_list, [1,-1,0,0], m_list)\
                                                        -co(s,0)*co(sprime,2)*integral4(l_list, [1,-1,0,2], m_list)\
                                                        -co(s,2)*co(sprime,0)*integral4(l_list, [1,-1,2,0], m_list)\
                                                        -co(s,2)*co(sprime,2)*integral4(l_list, [1,-1,2,2], m_list)\
                                                        +co(s,0)*co(sprime,0)*integral4(l_list, [1,1,0,0], m_list)\
                                                        +co(s,0)*co(sprime,2)*integral4(l_list, [1,1,0,2], m_list)\
                                                        +co(s,2)*co(sprime,0)*integral4(l_list, [1,1,2,0], m_list)\
                                                        +co(s,2)*co(sprime,2)*integral4(l_list, [1,1,2,2], m_list))

    return float(s9)

def S10(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s10=1.0/16*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(co(l,2)*co(s,0)*integral4(l_list, [-1,-2,0,1], m_list)\
                                                        +co(l,2)*co(s,2)*integral4(l_list, [-1,-2,2,1], m_list)\
                                                        -2*co(l,0)*co(s,0)*integral4(l_list, [-1,0,0,1], m_list)\
                                                        -2*co(l,0)*co(s,2)*integral4(l_list, [-1,0,2,1], m_list)\
                                                        +co(l,2)*co(s,0)*integral4(l_list, [-1,2,0,1], m_list)\
                                                        +co(l,2)*co(s,2)*integral4(l_list, [-1,2,2,1], m_list)\
                                                        -co(l,2)*co(s,0)*integral4(l_list, [1,-2,0,1], m_list)\
                                                        -co(l,2)*co(s,2)*integral4(l_list, [1,-2,2,1], m_list)\
                                                        +2*co(l,0)*co(s,0)*integral4(l_list, [1,0,0,1], m_list)\
                                                        +2*co(l,0)*co(s,2)*integral4(l_list, [1,0,2,1], m_list)\
                                                        -co(l,2)*co(s,0)*integral4(l_list, [1,2,0,1], m_list)\
                                                        -co(l,2)*co(s,2)*integral4(l_list, [1,2,2,1], m_list))

    return float(s10)

def S11(lprime,l,s,sprime,mprime,m):
    l_list5=[lprime,l,s,sprime,1]
    m_list5=[mprime,m,0,0,0]
    K1=co(sprime,2)*co(sprime,2)-2*co(sprime,0)*co(sprime,0)
    
    #The indentation doesnt matter for the line break \
    s11=m/8.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(np.sqrt(np.pi/3)*(co(s,0)*co(sprime,0)*integral5(l_list5,[-1,-1,0,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[-1,-1,2,0,0],m_list5)\
                                                                           +co(s,0)*co(sprime,0)*integral5(l_list5,[-1,1,0,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[-1,1,2,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[1,-1,0,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[1,-1,2,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[1,1,0,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[1,1,2,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[-1,-1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[-1,-1,2,2,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[-1,1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[-1,1,2,2,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[1,-1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[1,-1,2,2,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[1,1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[1,1,2,2,0],m_list5))\
                                                         -np.sqrt(np.pi/6)*(co(s,0)*K1*integral5(l_list5,[-1,-1,0,1,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[-1,-1,2,1,1],m_list5)\
                                                                           +co(s,0)*K1*integral5(l_list5,[-1,1,0,1,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[-1,1,2,1,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[1,-1,0,1,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[1,-1,2,1,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[1,1,0,1,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[1,1,2,1,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,0,3,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,2,3,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,0,3,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,2,3,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,0,3,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,2,3,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,0,3,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,2,3,1],m_list5)))
    return float(s11)


def S12(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s12=1/8.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(co(s,0)*co(sprime,0)*integral4(l_list, [-1,-1,0,0], m_list)\
                                                         +co(s,0)*co(sprime,2)*integral4(l_list, [-1,-1,0,2], m_list)\
                                                         -co(s,2)*co(sprime,0)*integral4(l_list, [-1,-1,2,0], m_list)\
                                                         -co(s,2)*co(sprime,2)*integral4(l_list, [-1,-1,2,2], m_list)\
                                                         -co(s,0)*co(sprime,0)*integral4(l_list, [-1,1,0,0], m_list)\
                                                         -co(s,0)*co(sprime,2)*integral4(l_list, [-1,1,0,2], m_list)\
                                                         +co(s,2)*co(sprime,0)*integral4(l_list, [-1,1,2,0], m_list)\
                                                         +co(s,2)*co(sprime,2)*integral4(l_list, [-1,1,2,2], m_list)\
                                                         -co(s,0)*co(sprime,0)*integral4(l_list, [1,-1,0,0], m_list)\
                                                         -co(s,0)*co(sprime,2)*integral4(l_list, [1,-1,0,2], m_list)\
                                                         +co(s,2)*co(sprime,0)*integral4(l_list, [1,-1,2,0], m_list)\
                                                         +co(s,2)*co(sprime,2)*integral4(l_list, [1,-1,2,2], m_list)\
                                                         +co(s,0)*co(sprime,0)*integral4(l_list, [1,1,0,0], m_list)\
                                                         +co(s,0)*co(sprime,2)*integral4(l_list, [1,1,0,2], m_list)\
                                                         -co(s,2)*co(sprime,0)*integral4(l_list, [1,1,2,0], m_list)\
                                                         -co(s,2)*co(sprime,2)*integral4(l_list, [1,1,2,2], m_list))

    return float(s12)


def S13(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s13=1/16.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(co(l,2)*co(s,0)*integral4(l_list, [-1,-2,0,1], m_list)\
                                                          -co(l,2)*co(s,2)*integral4(l_list, [-1,-2,2,1], m_list)\
                                                          -2*co(l,0)*co(s,0)*integral4(l_list, [-1,0,0,1], m_list)\
                                                          +2*co(l,0)*co(s,2)*integral4(l_list, [-1,0,2,1], m_list)\
                                                          +co(l,2)*co(s,0)*integral4(l_list, [-1,2,0,1], m_list)\
                                                          -co(l,2)*co(s,2)*integral4(l_list, [-1,2,2,1], m_list)\
                                                          -co(l,2)*co(s,0)*integral4(l_list, [1,-2,0,1], m_list)\
                                                          +co(l,2)*co(s,2)*integral4(l_list, [1,-2,2,1], m_list)\
                                                          +2*co(l,0)*co(s,0)*integral4(l_list, [1,0,0,1], m_list)\
                                                          -2*co(l,0)*co(s,2)*integral4(l_list, [1,0,2,1], m_list)\
                                                          -co(l,2)*co(s,0)*integral4(l_list, [1,2,0,1], m_list)\
                                                          +co(l,2)*co(s,2)*integral4(l_list, [1,2,2,1], m_list))    

    return float(s13)

def S14(lprime,l,s,sprime,mprime,m):
    l_list5=[lprime,l,s,sprime,1]
    m_list5=[mprime,m,0,0,0]
    K1=co(sprime,2)*co(sprime,2)-2*co(sprime,0)*co(sprime,0)

    s14=m/8.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(np.sqrt(np.pi/3)*(co(s,0)*co(sprime,0)*integral5(l_list5,[-1,-1,0,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[-1,-1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[-1,-1,2,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[-1,-1,2,2,0],m_list5)\
                                                                           +co(s,0)*co(sprime,0)*integral5(l_list5,[-1,1,0,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[-1,1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[-1,1,2,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[-1,1,2,2,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[1,-1,0,0,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[1,-1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[1,-1,2,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[1,-1,2,2,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[1,1,0,0,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[1,1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[1,1,2,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[1,1,2,2,0],m_list5))\
                                                         -np.sqrt(np.pi/6)*(co(s,0)*K1*integral5(l_list5,[-1,-1,0,1,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,0,3,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[-1,-1,2,1,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,2,3,1],m_list5)\
                                                                           +co(s,0)*K1*integral5(l_list5,[-1,1,0,1,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,0,3,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[-1,1,2,1,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,2,3,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[1,-1,0,1,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,0,3,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[1,-1,2,1,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,2,3,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[1,1,0,1,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,0,3,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[1,1,2,1,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,2,3,1],m_list5)))

    return float(s14)


def S15(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s15=1/16.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(co(s,0)*co(sprime,0)*integral4(l_list, [-1,-1,0,0], m_list)\
                                                          -co(s,0)*co(sprime,2)*integral4(l_list, [-1,-1,0,2], m_list)\
                                                          +co(s,2)*co(sprime,0)*integral4(l_list, [-1,-1,2,0], m_list)\
                                                          -co(s,2)*co(sprime,2)*integral4(l_list, [-1,-1,2,2], m_list)\
                                                          -co(s,0)*co(sprime,0)*integral4(l_list, [-1,1,0,0], m_list)\
                                                          +co(s,0)*co(sprime,2)*integral4(l_list, [-1,1,0,2], m_list)\
                                                          -co(s,2)*co(sprime,0)*integral4(l_list, [-1,1,2,0], m_list)\
                                                          +co(s,2)*co(sprime,2)*integral4(l_list, [-1,1,2,2], m_list)\
                                                          -co(s,0)*co(sprime,0)*integral4(l_list, [1,-1,0,0], m_list)\
                                                          +co(s,0)*co(sprime,2)*integral4(l_list, [1,-1,0,2], m_list)\
                                                          -co(s,2)*co(sprime,0)*integral4(l_list, [1,-1,2,0], m_list)\
                                                          +co(s,2)*co(sprime,2)*integral4(l_list, [1,-1,2,2], m_list)\
                                                          +co(s,0)*co(sprime,0)*integral4(l_list, [1,1,0,0], m_list)\
                                                          -co(s,0)*co(sprime,2)*integral4(l_list, [1,1,0,2], m_list)\
                                                          +co(s,2)*co(sprime,0)*integral4(l_list, [1,1,2,0], m_list)\
                                                          -co(s,2)*co(sprime,2)*integral4(l_list, [1,1,2,2], m_list))

    return float(s15)


def S16(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s16=1/16.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(co(s,0)*co(sprime,0)*integral4(l_list, [-1,-1,0,0], m_list)\
                                                          -co(s,0)*co(sprime,2)*integral4(l_list, [-1,-1,0,2], m_list)\
                                                          -co(s,2)*co(sprime,0)*integral4(l_list, [-1,-1,2,0], m_list)\
                                                          +co(s,2)*co(sprime,2)*integral4(l_list, [-1,-1,2,2], m_list)\
                                                          -co(s,0)*co(sprime,0)*integral4(l_list, [-1,1,0,0], m_list)\
                                                          +co(s,0)*co(sprime,2)*integral4(l_list, [-1,1,0,2], m_list)\
                                                          +co(s,2)*co(sprime,0)*integral4(l_list, [-1,1,2,0], m_list)\
                                                          -co(s,2)*co(sprime,2)*integral4(l_list, [-1,1,2,2], m_list)\
                                                          -co(s,0)*co(sprime,0)*integral4(l_list, [1,-1,0,0], m_list)\
                                                          +co(s,0)*co(sprime,2)*integral4(l_list, [1,-1,0,2], m_list)\
                                                          +co(s,2)*co(sprime,0)*integral4(l_list, [1,-1,2,0], m_list)\
                                                          -co(s,2)*co(sprime,2)*integral4(l_list, [1,-1,2,2], m_list)\
                                                          +co(s,0)*co(sprime,0)*integral4(l_list, [1,1,0,0], m_list)\
                                                          -co(s,0)*co(sprime,2)*integral4(l_list, [1,1,0,2], m_list)\
                                                          -co(s,2)*co(sprime,0)*integral4(l_list, [1,1,2,0], m_list)\
                                                          +co(s,2)*co(sprime,2)*integral4(l_list, [1,1,2,2], m_list))

    return float(s16)


def S17(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s17=1/4.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(integral4(l_list, [-1,-1,1,1], m_list)\
                                                         -integral4(l_list, [-1,1,1,1], m_list)\
                                                         -integral4(l_list, [1,-1,1,1], m_list)\
                                                         +integral4(l_list, [1,1,1,1], m_list))

    return float(s17)

def S18(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]
    K2=co(l,2)*co(l,2)+2*co(l,0)*co(l,0)
    
    s18=1/16.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(co(l,2)*co(l,3)*integral4(l_list, [-1,-3,1,1], m_list)\
                                                          -K2*integral4(l_list, [-1,-1,1,1], m_list)\
                                                          +K2*integral4(l_list, [-1,1,1,1], m_list)\
                                                          -co(l,2)*co(l,3)*integral4(l_list, [-1,3,1,1], m_list)\
                                                          -co(l,2)*co(l,3)*integral4(l_list, [1,-3,1,1], m_list)\
                                                          +K2*integral4(l_list, [1,-1,1,1], m_list)\
                                                          -K2*integral4(l_list, [1,1,1,1], m_list)\
                                                          +co(l,2)*co(l,3)*integral4(l_list, [1,3,1,1], m_list))
    

    return float(s18)


def S19(lprime,l,s,sprime,mprime,m):
    l_list6=[lprime,l,s,1,sprime,1]
    m_list6=[mprime,m,0,0,0,0]
    K1=co(sprime,2)*co(sprime,2)-2*co(sprime,0)*co(sprime,0)
    K2=co(s,2)*co(s,2)-2*co(s,0)*co(s,0)
    
    s19=-m**2/4.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(np.sqrt(np.pi/3)*(co(s,0)*co(sprime,0)*integral6(l_list6,[-1,-1,0,0,0,0],m_list6)\
                                                                               -co(s,0)*co(sprime,2)*integral6(l_list6,[-1,-1,0,0,2,0],m_list6)\
                                                                               -co(s,2)*co(sprime,0)*integral6(l_list6,[-1,-1,2,0,0,0],m_list6)\
                                                                               +co(s,2)*co(sprime,2)*integral6(l_list6,[-1,-1,2,0,2,0],m_list6)\
                                                                               -co(s,0)*co(sprime,0)*integral6(l_list6,[-1,1,0,0,0,0],m_list6)\
                                                                               +co(s,0)*co(sprime,2)*integral6(l_list6,[-1,1,0,0,2,0],m_list6)\
                                                                               +co(s,2)*co(sprime,0)*integral6(l_list6,[-1,1,2,0,0,0],m_list6)\
                                                                               -co(s,2)*co(sprime,2)*integral6(l_list6,[-1,1,2,0,2,0],m_list6)\
                                                                               -co(s,0)*co(sprime,0)*integral6(l_list6,[1,-1,0,0,0,0],m_list6)\
                                                                               +co(s,0)*co(sprime,2)*integral6(l_list6,[1,-1,0,0,2,0],m_list6)\
                                                                               +co(s,2)*co(sprime,0)*integral6(l_list6,[1,-1,2,0,0,0],m_list6)\
                                                                               -co(s,2)*co(sprime,2)*integral6(l_list6,[1,-1,2,0,2,0],m_list6)\
                                                                               +co(s,0)*co(sprime,0)*integral6(l_list6,[1,1,0,0,0,0],m_list6)\
                                                                               -co(s,0)*co(sprime,2)*integral6(l_list6,[1,1,0,0,2,0],m_list6)\
                                                                               -co(s,2)*co(sprime,0)*integral6(l_list6,[1,1,2,0,0,0],m_list6)\
                                                                               +co(s,2)*co(sprime,2)*integral6(l_list6,[1,1,2,0,2,0],m_list6))\
                                                                               -np.pi/(3*np.sqrt(2))*(K2*co(sprime,0)*integral6(l_list6,[-1,-1,1,1,0,0],m_list6)\
                                                                               -K2*co(sprime,2)*integral6(l_list6,[-1,-1,1,1,2,0],m_list6)\
                                                                               -co(s,2)*co(s,3)*co(sprime,0)*integral6(l_list6,[-1,-1,3,1,0,0],m_list6)\
                                                                               +co(s,2)*co(s,3)*co(sprime,2)*integral6(l_list6,[-1,-1,3,1,2,0],m_list6)\
                                                                               -K2*co(sprime,0)*integral6(l_list6,[-1,1,1,1,0,0],m_list6)\
                                                                               +K2*co(sprime,2)*integral6(l_list6,[-1,1,1,1,2,0],m_list6)\
                                                                               +co(s,2)*co(s,3)*co(sprime,0)*integral6(l_list6,[-1,1,3,1,0,0],m_list6)\
                                                                               -co(s,2)*co(s,3)*co(sprime,2)*integral6(l_list6,[-1,1,3,1,2,0],m_list6)\
                                                                               -K2*co(sprime,0)*integral6(l_list6,[1,-1,1,1,0,0],m_list6)\
                                                                               +K2*co(sprime,2)*integral6(l_list6,[1,-1,1,1,2,0],m_list6)\
                                                                               +co(s,2)*co(s,3)*co(sprime,0)*integral6(l_list6,[1,-1,3,1,0,0],m_list6)\
                                                                               -co(s,2)*co(s,3)*co(sprime,2)*integral6(l_list6,[1,-1,3,1,2,0],m_list6)\
                                                                               +K2*co(sprime,0)*integral6(l_list6,[1,1,1,1,0,0],m_list6)\
                                                                               -K2*co(sprime,2)*integral6(l_list6,[1,1,1,1,2,0],m_list6)\
                                                                               -co(s,2)*co(s,3)*co(sprime,0)*integral6(l_list6,[1,1,3,1,0,0],m_list6)\
                                                                               +co(s,2)*co(s,3)*co(sprime,2)*integral6(l_list6,[1,1,3,1,2,0],m_list6)\
                                                                               +co(s,0)*K1*integral6(l_list6,[-1,-1,0,0,1,1],m_list6)\
                                                                               -co(s,0)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,-1,0,0,3,1],m_list6)\
                                                                               -co(s,2)*K1*integral6(l_list6,[-1,-1,2,0,1,1],m_list6)\
                                                                               +co(s,2)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,-1,2,0,3,1],m_list6)\
                                                                               -co(s,0)*K1*integral6(l_list6,[-1,1,0,0,1,1],m_list6)\
                                                                               +co(s,0)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,1,0,0,3,1],m_list6)\
                                                                               +co(s,2)*K1*integral6(l_list6,[-1,1,2,0,1,1],m_list6)\
                                                                               -co(s,2)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,1,2,0,3,1],m_list6)\
                                                                               -co(s,0)*K1*integral6(l_list6,[1,-1,0,0,1,1],m_list6)\
                                                                               +co(s,0)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,-1,0,0,3,1],m_list6)\
                                                                               +co(s,2)*K1*integral6(l_list6,[1,-1,2,0,1,1],m_list6)\
                                                                               -co(s,2)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,-1,2,0,3,1],m_list6)\
                                                                               +co(s,0)*K1*integral6(l_list6,[1,1,0,0,1,1],m_list6)\
                                                                               -co(s,0)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,1,0,0,3,1],m_list6)\
                                                                               -co(s,2)*K1*integral6(l_list6,[1,1,2,0,1,1],m_list6)\
                                                                               +co(s,2)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,1,2,0,3,1],m_list6))\
                                                                               +np.pi/6*(K2*K1*integral6(l_list6,[-1,-1,1,1,1,1],m_list6)\
                                                                               -K2*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,-1,1,1,3,1],m_list6)\
                                                                               -co(s,2)*co(s,3)*K1*integral6(l_list6,[-1,-1,3,1,1,1],m_list6)\
                                                                               +co(s,2)*co(s,3)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,-1,3,1,3,1],m_list6)\
                                                                               -K2*K1*integral6(l_list6,[-1,1,1,1,1,1],m_list6)\
                                                                               +K2*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,1,1,1,3,1],m_list6)\
                                                                               +co(s,2)*co(s,3)*K1*integral6(l_list6,[-1,1,3,1,1,1],m_list6)\
                                                                               -co(s,2)*co(s,3)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[-1,1,3,1,3,1],m_list6)\
                                                                               -K2*K1*integral6(l_list6,[1,-1,1,1,1,1],m_list6)\
                                                                               +K2*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,-1,1,1,3,1],m_list6)\
                                                                               +co(s,2)*co(s,3)*K1*integral6(l_list6,[1,-1,3,1,1,1],m_list6)\
                                                                               -co(s,2)*co(s,3)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,-1,3,1,3,1],m_list6)\
                                                                               +K2*K1*integral6(l_list6,[1,1,1,1,1,1],m_list6)\
                                                                               -K2*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,1,1,1,3,1],m_list6)\
                                                                               -co(s,2)*co(s,3)*K1*integral6(l_list6,[1,1,3,1,1,1],m_list6)\
                                                                               +co(s,2)*co(s,3)*co(sprime,2)*co(sprime,3)*integral6(l_list6,[1,1,3,1,3,1],m_list6)))
    return float(s19)

def S20(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]   
    K4=co(s,2)*co(s,2)+2*co(s,0)*co(s,0)
    
    s20=-1/16.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(K4*integral4(l_list, [-1,-1,1,1], m_list)\
                                                           -co(s,2)*co(s,3)*integral4(l_list, [-1,-1,3,1], m_list)\
                                                           -K4*integral4(l_list, [-1,1,1,1], m_list)\
                                                           +co(s,2)*co(s,3)*integral4(l_list, [-1,1,3,1], m_list)\
                                                           -K4*integral4(l_list, [1,-1,1,1], m_list)\
                                                           +co(s,2)*co(s,3)*integral4(l_list, [1,-1,3,1], m_list)\
                                                           +K4*integral4(l_list, [1,1,1,1], m_list)\
                                                           -co(s,2)*co(s,3)*integral4(l_list, [1,1,3,1], m_list))

    return float(s20)


def S21(lprime,l,s,sprime,mprime,m):
    l_list=[lprime,l,s,sprime]
    m_list=[mprime,m,0,0]

    s21=1/4.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(integral4(l_list, [-1,-1,1,1], m_list)\
                                                         +integral4(l_list, [-1,1,1,1], m_list)\
                                                         +integral4(l_list, [1,-1,1,1], m_list)\
                                                         +integral4(l_list, [1,1,1,1], m_list))

    return float(s21)


def S22(lprime,l,s,sprime,mprime,m):
    l_list5=[lprime,l,s,sprime,1]
    m_list5=[mprime,m,0,0,0]
    K1=co(sprime,2)*co(sprime,2)-2*co(sprime,0)*co(sprime,0)

    s22=-m/8.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(np.sqrt(np.pi/3)*(co(s,0)*co(sprime,0)*integral5(l_list5,[-1,-1,0,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[-1,-1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[-1,-1,2,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[-1,-1,2,2,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[-1,1,0,0,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[-1,1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[-1,1,2,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[-1,1,2,2,0],m_list5)\
                                                                           +co(s,0)*co(sprime,0)*integral5(l_list5,[1,-1,0,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[1,-1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[1,-1,2,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[1,-1,2,2,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[1,1,0,0,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[1,1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[1,1,2,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[1,1,2,2,0],m_list5))\
                                                         -np.sqrt(np.pi/6)*(co(s,0)*K1*integral5(l_list5,[-1,-1,0,1,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,0,3,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[-1,-1,2,1,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,2,3,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[-1,1,0,1,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,0,3,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[-1,1,2,1,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,2,3,1],m_list5)\
                                                                           +co(s,0)*K1*integral5(l_list5,[1,-1,0,1,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,0,3,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[1,-1,2,1,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,2,3,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[1,1,0,1,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,0,3,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[1,1,2,1,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,2,3,1],m_list5)))

    return float(s22)

#S23=-S14? or only for s=s'?
def S23(lprime,l,s,sprime,mprime,m):
    l_list5=[lprime,l,s,sprime,1]
    m_list5=[mprime,m,0,0,0]
    K1=co(sprime,2)*co(sprime,2)-2*co(sprime,0)*co(sprime,0)

    s23=-m/8.0*co(lprime,0)*co(l,0)*co(s,0)*co(sprime,0)*(np.sqrt(np.pi/3)*(co(s,0)*co(sprime,0)*integral5(l_list5,[-1,-1,0,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[-1,-1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[-1,-1,2,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[-1,-1,2,2,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[-1,1,0,0,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[-1,1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[-1,1,2,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[-1,1,2,2,0],m_list5)\
                                                                           +co(s,0)*co(sprime,0)*integral5(l_list5,[1,-1,0,0,0],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*integral5(l_list5,[1,-1,0,2,0],m_list5)\
                                                                           -co(s,2)*co(sprime,0)*integral5(l_list5,[1,-1,2,0,0],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*integral5(l_list5,[1,-1,2,2,0],m_list5)\
                                                                           -co(s,0)*co(sprime,0)*integral5(l_list5,[1,1,0,0,0],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*integral5(l_list5,[1,1,0,2,0],m_list5)\
                                                                           +co(s,2)*co(sprime,0)*integral5(l_list5,[1,1,2,0,0],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*integral5(l_list5,[1,1,2,2,0],m_list5))\
                                                         -np.sqrt(np.pi/6)*(co(s,0)*K1*integral5(l_list5,[-1,-1,0,1,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,0,3,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[-1,-1,2,1,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,-1,2,3,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[-1,1,0,1,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,0,3,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[-1,1,2,1,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[-1,1,2,3,1],m_list5)\
                                                                           +co(s,0)*K1*integral5(l_list5,[1,-1,0,1,1],m_list5)\
                                                                           -co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,0,3,1],m_list5)\
                                                                           -co(s,2)*K1*integral5(l_list5,[1,-1,2,1,1],m_list5)\
                                                                           +co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,-1,2,3,1],m_list5)\
                                                                           -co(s,0)*K1*integral5(l_list5,[1,1,0,1,1],m_list5)\
                                                                           +co(s,0)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,0,3,1],m_list5)\
                                                                           +co(s,2)*K1*integral5(l_list5,[1,1,2,1,1],m_list5)\
                                                                           -co(s,2)*co(sprime,2)*co(sprime,3)*integral5(l_list5,[1,1,2,3,1],m_list5)))

    return float(s23)

def main():
    '''
    start_time = time.time()

    # TEST_AREA
    s = 2
    sprime = 2
    lprime = 4
    l = 4
    m = -4
    l_list = [lprime, l, s, sprime]
    m_list = [m, m, 0, 0]

    print('For [lprime,l,s,sprime,m=mprime]=[', lprime, l, s, sprime, m, ']')

    print('S1=', S1(lprime, l, s, sprime, m, m))
    print('S2=', S2(lprime, l, s, sprime, m, m))
    print('S3=', S3(lprime, l, s, sprime, m, m))
    print('S4=', S4(lprime, l, s, sprime, m, m))
    print('S5=', S5(lprime, l, s, sprime, m, m))
    print('S6=', S6(lprime, l, s, sprime, m, m))
    print('S7=', S7(lprime, l, s, sprime, m, m))
    print('S8=', S8(lprime, l, s, sprime, m, m))
    print('S9=', S9(lprime, l, s, sprime, m, m))
    print('S10=', S10(lprime, l, s, sprime, m, m))
    print('S11=', S11(lprime, l, s, sprime, m, m))
    print('S12=', S12(lprime, l, s, sprime, m, m))
    print('S13=', S13(lprime, l, s, sprime, m, m))
    print('S14=', S14(lprime, l, s, sprime, m, m))
    print('S15=', S15(lprime, l, s, sprime, m, m))
    print('S16=', S16(lprime, l, s, sprime, m, m))
    print('S17=', S17(lprime, l, s, sprime, m, m))
    print('S18=', S18(lprime, l, s, sprime, m, m))
    print('S19=', S19(lprime, l, s, sprime, m, m))
    print('S20=', S20(lprime, l, s, sprime, m, m))
    print('S21=', S21(lprime, l, s, sprime, m, m))
    print('S22=', S22(lprime, l, s, sprime, m, m))
    print('S23=', S23(lprime, l, s, sprime, m, m))

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    '''

if __name__ == '__main__':
    main()
