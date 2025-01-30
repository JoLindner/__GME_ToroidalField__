from Integral_over_gsh import integral_over_4gsh as integral4
from Integral_over_gsh import integral_over_5gsh as integral5
from Integral_over_gsh import integral_over_6gsh as integral6
import numpy as np
from constants import c_const as co
import time


class AngularKernels:
    def __init__(self, lprime, l, s, sprime, mprime, m):
        # Load parameters
        self.lprime = lprime
        self.l = l
        self.s = s
        self.sprime = sprime
        self.mprime = mprime
        self.m = m
        self.l_list = [lprime, l, s, sprime]
        self.m_list = [mprime, m, 0, 0]
        self.l_list5 = [lprime, l, s, sprime, 1]
        self.m_list5 = [mprime, m, 0, 0, 0]
        self.l_list6 = [lprime, l, s, 1, sprime, 1]
        self.m_list6 = [mprime, m, 0, 0, 0, 0]

        # Precompute common constants
        self.co_s_0 = co(s, 0)
        self.co_sprime_0 = co(sprime, 0)
        self.co_l_0 = co(l, 0)
        self.co_lprime_0 = co(lprime, 0)
        self.co_s_2 = co(s, 2)
        self.co_sprime_2 = co(sprime, 2)
        self.co_l_2 = co(l, 2)
        self.co_s_3 = co(s, 3)
        self.co_sprime_3 = co(sprime, 3)
        self.co_l_3 = co(l, 3)
        
        self.K1 = self.co_sprime_2*self.co_sprime_2-2*self.co_sprime_0*self.co_sprime_0
        self.K2 = self.co_s_2*self.co_s_2-2*self.co_s_0*self.co_s_0
        self.K3 = self.co_l_2*self.co_l_2+2*self.co_l_0*self.co_l_0
        self.K4 = self.co_s_2*self.co_s_2+2*self.co_s_0*self.co_s_0


        # Dictionaries to store precomputed integrals
        self.integrals4 = {}
        self.integral4_definitions = [
            (0, -1, 0, 1),  # S2, S6
            (0, 1, 0, 1),   # S2, S6
            (0, -1, 2, 1),  # S2, S6
            (0, 1, 2, 1),   # S2, S6
            (-1, -1, 1, 1),  # S4, S17, S18, S20, S21
            (-1, 1, 1, 1),  # S4, S17, S18, S20, S21
            (1, -1, 1, 1),  # S4, S17, S18, S20, S21
            (1, 1, 1, 1),  # S4, S17, S18, S20, S21
            (-1, 0, 0, 1),  # S7, S8, S10, S13
            (-1, 0, 2, 1),  # S7, S8, S10, S13
            (1, 0, 0, 1),  # S7, S8, S10, S13
            (1, 0, 2, 1),  # S7, S8, S10, S13
            (-1, -1, 0, 0),  # S9, S12, S15, S16
            (-1, -1, 0, 2),  # S9, S12, S15, S16
            (-1, -1, 2, 0),  # S9, S12, S15, S16
            (-1, -1, 2, 2),  # S9, S12, S15, S16
            (-1, 1, 0, 0),  # S9, S12, S15, S16
            (-1, 1, 0, 2),  # S9, S12, S15, S16
            (-1, 1, 2, 0),  # S9, S12, S15, S16
            (-1, 1, 2, 2),  # S9, S12, S15, S16
            (1, -1, 0, 0),  # S9, S12, S15, S16
            (1, -1, 0, 2),  # S9, S12, S15, S16
            (1, -1, 2, 0),  # S9, S12, S15, S16
            (1, -1, 2, 2),  # S9, S12, S15, S16
            (1, 1, 0, 0),  # S9, S12, S15, S16
            (1, 1, 0, 2),  # S9, S12, S15, S16
            (1, 1, 2, 0),  # S9, S12, S15, S16
            (1, 1, 2, 2),  # S9, S12, S15, S16
            (-1, -2, 0, 1),  # S10, S13
            (-1, -2, 2, 1),  # S10, S13
            (-1, 2, 0, 1),  # S10, S13
            (-1, 2, 2, 1),  # S10, S13
            (1, -2, 0, 1),  # S10, S13
            (1, -2, 2, 1),  # S10, S13
            (1, 2, 0, 1),  # S10, S13
            (1, 2, 2, 1),  # S10, S13
        ]
        self.integrals5 = {}
        self.integral5_definitions = [
            (-1, -1, 0, 0, 0),  # S11, S14, S22, S23
            (-1, -1, 2, 0, 0),  # S11, S14, S22, S23
            (-1, 1, 0, 0, 0),  # S11, S14, S22, S23
            (-1, 1, 2, 0, 0),  # S11, S14, S22, S23
            (1, -1, 0, 0, 0),  # S11, S14, S22, S23
            (1, -1, 2, 0, 0),  # S11, S14, S22, S23
            (1, 1, 0, 0, 0),  # S11, S14, S22, S23
            (1, 1, 2, 0, 0),  # S11, S14, S22, S23
            (-1, -1, 0, 2, 0),  # S11, S14, S22, S23
            (-1, -1, 2, 2, 0),  # S11, S14, S22, S23
            (-1, 1, 0, 2, 0),  # S11, S14, S22, S23
            (-1, 1, 2, 2, 0),  # S11, S14, S22, S23
            (1, -1, 0, 2, 0),  # S11, S14, S22, S23
            (1, -1, 2, 2, 0),  # S11, S14, S22, S23
            (1, 1, 0, 2, 0),  # S11, S14, S22, S23
            (1, 1, 2, 2, 0),  # S11, S14, S22, S23
            (-1, -1, 0, 1, 1),  # S11, S14, S22, S23
            (-1, -1, 2, 1, 1),  # S11, S14, S22, S23
            (-1, 1, 0, 1, 1),  # S11, S14, S22, S23
            (-1, 1, 2, 1, 1),  # S11, S14, S22, S23
            (1, -1, 0, 1, 1),  # S11, S14, S22, S23
            (1, -1, 2, 1, 1),  # S11, S14, S22, S23
            (1, 1, 0, 1, 1),  # S11, S14, S22, S23
            (1, 1, 2, 1, 1),  # S11, S14, S22, S23
            (-1, -1, 0, 3, 1),  # S11, S14, S22, S23
            (-1, -1, 2, 3, 1),  # S11, S14, S22, S23
            (-1, 1, 0, 3, 1),  # S11, S14, S22, S23
            (-1, 1, 2, 3, 1),  # S11, S14, S22, S23
            (1, -1, 0, 3, 1),  # S11, S14, S22, S23
            (1, -1, 2, 3, 1),  # S11, S14, S22, S23
            (1, 1, 0, 3, 1),  # S11, S14, S22, S23
            (1, 1, 2, 3, 1),  # S11, S14, S22, S23
        ]

        # Precompute all redundant integrals
        self.precompute_integrals()

        # Compute kernels
        for kernel_name in [f"S{i}" for i in range(1, 24)]:
            getattr(self, kernel_name)()  

    def precompute_integrals(self):
        # Compute integrals over 4 gsh
        for key in self.integral4_definitions:
            self.integrals4[key] = integral4(self.l_list, list(key), self.m_list)
        # Compute integrals over 5 gsh
        for key in self.integral5_definitions:
            self.integrals5[key] = integral5(self.l_list5, list(key), self.m_list5)

    def S1(self):
        s1 = self.co_s_0*self.co_sprime_0*integral4(self.l_list,[0,0,1,1],self.m_list)
            
        self.s1 = float(s1)
    
    def S2(self):
        s2 = 1.0/4*self.co_s_0*self.co_sprime_0*self.co_l_0*(self.co_s_0*self.integrals4[(0, -1, 0, 1)]\
                                                            -self.co_s_0*self.integrals4[(0, 1, 0, 1)]\
                                                            +self.co_s_2*self.integrals4[(0, -1, 2, 1)]\
                                                            -self.co_s_2*self.integrals4[(0, 1, 2, 1)])
        self.s2 = float(s2)
      
    def S3(self):
        s3 = 1.0/4*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_l_2*integral4(self.l_list, [0,2,1,1], self.m_list)\
                                               +self.co_l_2*integral4(self.l_list, [0,-2,1,1], self.m_list)\
                                               -2*self.co_l_0*integral4(self.l_list, [0,0,1,1], self.m_list))
        self.s3 = float(s3)
    
    def S4(self):
        s4 = -1.0/4*self.co_l_0*self.co_lprime_0*self.co_s_0*self.co_sprime_0*(self.integrals4[(-1, -1, 1, 1)]\
                                                             +self.integrals4[(-1, 1, 1, 1)]\
                                                             +self.integrals4[(1, -1, 1, 1)]\
                                                             +self.integrals4[(1, 1, 1, 1)])
        self.s4 = float(s4)
    
    def S5(self):
        s5 = 1.0/4*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_sprime_0*integral4(self.l_list, [0,-1,1,0], self.m_list)\
                                               -self.co_sprime_0*integral4(self.l_list, [0,1,1,0], self.m_list)\
                                               -self.co_sprime_2*integral4(self.l_list, [0,-1,1,2], self.m_list)\
                                               +self.co_sprime_2*integral4(self.l_list, [0,1,1,2], self.m_list))
        self.s5 = float(s5)
    
    def S6(self):
        s6 = 1.0/4*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_s_0*self.integrals4[(0, -1, 0, 1)]\
                                               -self.co_s_0*self.integrals4[(0, 1, 0, 1)]\
                                               -self.co_s_2*self.integrals4[(0, -1, 2, 1)]\
                                               +self.co_s_2*self.integrals4[(0, 1, 2, 1)])
        self.s6 = float(s6)
    
    def S7(self):
        s7 = 1.0/4*self.co_lprime_0*self.co_s_0*self.co_sprime_0*(self.co_s_0*self.integrals4[(-1, 0, 0, 1)]\
                                                            +self.co_s_2*self.integrals4[(-1, 0, 2, 1)]\
                                                            -self.co_s_0*self.integrals4[(1, 0, 0, 1)]\
                                                            -self.co_s_2*self.integrals4[(1, 0, 2, 1)])
        self.s7 = float(s7)
     
    def S8(self):
        s8 = 1.0/4*self.co_lprime_0*self.co_s_0*self.co_sprime_0*(self.integrals4[(-1, 0, 0, 1)]\
                                                    -self.integrals4[(-1, 0, 2, 1)]\
                                                    -self.integrals4[(1, 0, 0, 1)]\
                                                    +self.integrals4[(1, 0, 2, 1)])
        self.s8 = float(s8)
    
    def S9(self):
        s9 = 1.0/16*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_s_0*self.co_sprime_0*self.integrals4[(-1, -1, 0, 0)]\
                                                            +self.co_s_0*self.co_sprime_2*self.integrals4[(-1, -1, 0, 2)]\
                                                            +self.co_s_2*self.co_sprime_0*self.integrals4[(-1, -1, 2, 0)]\
                                                            +self.co_s_2*self.co_sprime_2*self.integrals4[(-1, -1, 2, 2)]\
                                                            -self.co_s_0*self.co_sprime_0*self.integrals4[(-1, 1, 0, 0)]\
                                                            -self.co_s_0*self.co_sprime_2*self.integrals4[(-1, 1, 0, 2)]\
                                                            -self.co_s_2*self.co_sprime_0*self.integrals4[(-1, 1, 2, 0)]\
                                                            -self.co_s_2*self.co_sprime_2*self.integrals4[(-1, 1, 2, 2)]\
                                                            -self.co_s_0*self.co_sprime_0*self.integrals4[(1, -1, 0, 0)]\
                                                            -self.co_s_0*self.co_sprime_2*self.integrals4[(1, -1, 0, 2)]\
                                                            -self.co_s_2*self.co_sprime_0*self.integrals4[(1, -1, 2, 0)]\
                                                            -self.co_s_2*self.co_sprime_2*self.integrals4[(1, -1, 2, 2)]\
                                                            +self.co_s_0*self.co_sprime_0*self.integrals4[(1, 1, 0, 0)]\
                                                            +self.co_s_0*self.co_sprime_2*self.integrals4[(1, 1, 0, 2)]\
                                                            +self.co_s_2*self.co_sprime_0*self.integrals4[(1, 1, 2, 0)]\
                                                            +self.co_s_2*self.co_sprime_2*self.integrals4[(1, 1, 2, 2)])
    
        self.s9 = float(s9)
    
    def S10(self):
        s10 = 1.0/16*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_l_2*self.co_s_0*self.integrals4[(-1, -2, 0, 1)]\
                                                            +self.co_l_2*self.co_s_2*self.integrals4[(-1, -2, 2, 1)]\
                                                            -2*self.co_l_0*self.co_s_0*self.integrals4[(-1, 0, 0, 1)]\
                                                            -2*self.co_l_0*self.co_s_2*self.integrals4[(-1, 0, 2, 1)]\
                                                            +self.co_l_2*self.co_s_0*self.integrals4[(-1, 2, 0, 1)]\
                                                            +self.co_l_2*self.co_s_2*self.integrals4[(-1, 2, 2, 1)]\
                                                            -self.co_l_2*self.co_s_0*self.integrals4[(1, -2, 0, 1)]\
                                                            -self.co_l_2*self.co_s_2*self.integrals4[(1, -2, 2, 1)]\
                                                            +2*self.co_l_0*self.co_s_0*self.integrals4[(1, 0, 0, 1)]\
                                                            +2*self.co_l_0*self.co_s_2*self.integrals4[(1, 0, 2, 1)]\
                                                            -self.co_l_2*self.co_s_0*self.integrals4[(1, 2, 0, 1)]\
                                                            -self.co_l_2*self.co_s_2*self.integrals4[(1, 2, 2, 1)])
    
        self.s10 = float(s10)
    
    def S11(self):        
        # The indentation doesn't matter for the line break \
        s11 = self.m/8.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(np.sqrt(np.pi/3)*(self.co_s_0*self.co_sprime_0*self.integrals5[(-1, -1, 0, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(-1, -1, 2, 0, 0)]\
                                                                               +self.co_s_0*self.co_sprime_0*self.integrals5[(-1, 1, 0, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(-1, 1, 2, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(1,- 1, 0, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(1, -1, 2, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(1, 1, 0, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(1, 1, 2, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(-1, -1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(-1, -1, 2, 2, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(-1, 1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(-1, 1, 2, 2, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(1, -1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(1, -1, 2, 2, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(1, 1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(1, 1, 2, 2, 0)])\
                                                             -np.sqrt(np.pi/6)*(self.co_s_0*self.K1*self.integrals5[(-1, -1, 0, 1, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(-1, -1, 2, 1, 1)]\
                                                                               +self.co_s_0*self.K1*self.integrals5[(-1, 1, 0, 1, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(-1, 1, 2, 1, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(1, -1, 0, 1, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(1, -1, 2, 1, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(1, 1, 0, 1, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(1, 1, 2, 1, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 2, 3, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 2, 3, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 2, 3, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 2, 3, 1)]))

        self.s11 = float(s11)
    
    def S12(self):
        s12 = 1/8.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_s_0*self.co_sprime_0*self.integrals4[(-1, -1, 0, 0)]\
                                                             +self.co_s_0*self.co_sprime_2*self.integrals4[(-1, -1, 0, 2)]\
                                                             -self.co_s_2*self.co_sprime_0*self.integrals4[(-1, -1, 2, 0)]\
                                                             -self.co_s_2*self.co_sprime_2*self.integrals4[(-1, -1, 2, 2)]\
                                                             -self.co_s_0*self.co_sprime_0*self.integrals4[(-1, 1, 0, 0)]\
                                                             -self.co_s_0*self.co_sprime_2*self.integrals4[(-1, 1, 0, 2)]\
                                                             +self.co_s_2*self.co_sprime_0*self.integrals4[(-1, 1, 2, 0)]\
                                                             +self.co_s_2*self.co_sprime_2*self.integrals4[(-1, 1, 2, 2)]\
                                                             -self.co_s_0*self.co_sprime_0*self.integrals4[(1, -1, 0, 0)]\
                                                             -self.co_s_0*self.co_sprime_2*self.integrals4[(1, -1, 0, 2)]\
                                                             +self.co_s_2*self.co_sprime_0*self.integrals4[(1, -1, 2, 0)]\
                                                             +self.co_s_2*self.co_sprime_2*self.integrals4[(1, -1, 2, 2)]\
                                                             +self.co_s_0*self.co_sprime_0*self.integrals4[(1, 1, 0, 0)]\
                                                             +self.co_s_0*self.co_sprime_2*self.integrals4[(1, 1, 0, 2)]\
                                                             -self.co_s_2*self.co_sprime_0*self.integrals4[(1, 1, 2, 0)]\
                                                             -self.co_s_2*self.co_sprime_2*self.integrals4[(1, 1, 2, 2)])
    
        self.s12 = float(s12)

    def S13(self):
        s13 = 1/16.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_l_2*self.co_s_0*self.integrals4[(-1, -2, 0, 1)]\
                                                              -self.co_l_2*self.co_s_2*self.integrals4[(-1, -2, 2, 1)]\
                                                              -2*self.co_l_0*self.co_s_0*self.integrals4[(-1, 0, 0, 1)]\
                                                              +2*self.co_l_0*self.co_s_2*self.integrals4[(-1, 0, 2, 1)]\
                                                              +self.co_l_2*self.co_s_0*self.integrals4[(-1, 2, 0, 1)]\
                                                              -self.co_l_2*self.co_s_2*self.integrals4[(-1, 2, 2, 1)]\
                                                              -self.co_l_2*self.co_s_0*self.integrals4[(1, -2, 0, 1)]\
                                                              +self.co_l_2*self.co_s_2*self.integrals4[(1, -2, 2, 1)]\
                                                              +2*self.co_l_0*self.co_s_0*self.integrals4[(1, 0, 0, 1)]\
                                                              -2*self.co_l_0*self.co_s_2*self.integrals4[(1, 0, 2, 1)]\
                                                              -self.co_l_2*self.co_s_0*self.integrals4[(1, 2, 0, 1)]\
                                                              +self.co_l_2*self.co_s_2*self.integrals4[(1, 2, 2, 1)])    
    
        self.s13 = float(s13)
    
    def S14(self):
        s14 = self.m/8.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(np.sqrt(np.pi/3)*(self.co_s_0*self.co_sprime_0*self.integrals5[(-1, -1, 0, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(-1, -1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(-1, -1, 2, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(-1, -1, 2, 2, 0)]\
                                                                               +self.co_s_0*self.co_sprime_0*self.integrals5[(-1, 1, 0, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(-1, 1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(-1, 1, 2, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(-1, 1, 2, 2, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(1,- 1, 0, 0, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(1, -1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(1, -1, 2, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(1, -1, 2, 2, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(1, 1, 0, 0, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(1, 1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(1, 1, 2, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(1, 1, 2, 2, 0)])\
                                                             -np.sqrt(np.pi/6)*(self.co_s_0*self.K1*self.integrals5[(-1, -1, 0, 1, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(-1, -1, 2, 1, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 2, 3, 1)]\
                                                                               +self.co_s_0*self.K1*self.integrals5[(-1, 1, 0, 1, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(-1, 1, 2, 1, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 2, 3, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(1, -1, 0, 1, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(1, -1, 2, 1, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 2, 3, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(1, 1, 0, 1, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(1, 1, 2, 1, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 2, 3, 1)]))
    
        self.s14 = float(s14)
      
    def S15(self):
        s15 = 1/16.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_s_0*self.co_sprime_0*self.integrals4[(-1, -1, 0, 0)]\
                                                              -self.co_s_0*self.co_sprime_2*self.integrals4[(-1, -1, 0, 2)]\
                                                              +self.co_s_2*self.co_sprime_0*self.integrals4[(-1, -1, 2, 0)]\
                                                              -self.co_s_2*self.co_sprime_2*self.integrals4[(-1, -1, 2, 2)]\
                                                              -self.co_s_0*self.co_sprime_0*self.integrals4[(-1, 1, 0, 0)]\
                                                              +self.co_s_0*self.co_sprime_2*self.integrals4[(-1, 1, 0, 2)]\
                                                              -self.co_s_2*self.co_sprime_0*self.integrals4[(-1, 1, 2, 0)]\
                                                              +self.co_s_2*self.co_sprime_2*self.integrals4[(-1, 1, 2, 2)]\
                                                              -self.co_s_0*self.co_sprime_0*self.integrals4[(1, -1, 0, 0)]\
                                                              +self.co_s_0*self.co_sprime_2*self.integrals4[(1, -1, 0, 2)]\
                                                              -self.co_s_2*self.co_sprime_0*self.integrals4[(1, -1, 2, 0)]\
                                                              +self.co_s_2*self.co_sprime_2*self.integrals4[(1, -1, 2, 2)]\
                                                              +self.co_s_0*self.co_sprime_0*self.integrals4[(1, 1, 0, 0)]\
                                                              -self.co_s_0*self.co_sprime_2*self.integrals4[(1, 1, 0, 2)]\
                                                              +self.co_s_2*self.co_sprime_0*self.integrals4[(1, 1, 2, 0)]\
                                                              -self.co_s_2*self.co_sprime_2*self.integrals4[(1, 1, 2, 2)])
    
        self.s15 = float(s15)
    
    def S16(self):
        s16 = 1/16.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_s_0*self.co_sprime_0*self.integrals4[(-1, -1, 0, 0)]\
                                                              -self.co_s_0*self.co_sprime_2*self.integrals4[(-1, -1, 0, 2)]\
                                                              -self.co_s_2*self.co_sprime_0*self.integrals4[(-1, -1, 2, 0)]\
                                                              +self.co_s_2*self.co_sprime_2*self.integrals4[(-1, -1, 2, 2)]\
                                                              -self.co_s_0*self.co_sprime_0*self.integrals4[(-1, 1, 0, 0)]\
                                                              +self.co_s_0*self.co_sprime_2*self.integrals4[(-1, 1, 0, 2)]\
                                                              +self.co_s_2*self.co_sprime_0*self.integrals4[(-1, 1, 2, 0)]\
                                                              -self.co_s_2*self.co_sprime_2*self.integrals4[(-1, 1, 2, 2)]\
                                                              -self.co_s_0*self.co_sprime_0*self.integrals4[(1, -1, 0, 0)]\
                                                              +self.co_s_0*self.co_sprime_2*self.integrals4[(1, -1, 0, 2)]\
                                                              +self.co_s_2*self.co_sprime_0*self.integrals4[(1, -1, 2, 0)]\
                                                              -self.co_s_2*self.co_sprime_2*self.integrals4[(1, -1, 2, 2)]\
                                                              +self.co_s_0*self.co_sprime_0*self.integrals4[(1, 1, 0, 0)]\
                                                              -self.co_s_0*self.co_sprime_2*self.integrals4[(1, 1, 0, 2)]\
                                                              -self.co_s_2*self.co_sprime_0*self.integrals4[(1, 1, 2, 0)]\
                                                              +self.co_s_2*self.co_sprime_2*self.integrals4[(1, 1, 2, 2)])
    
        self.s16 = float(s16)

    def S17(self):
        s17 = 1/4.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.integrals4[(-1, -1, 1, 1)]\
                                                             -self.integrals4[(-1, 1, 1, 1)]\
                                                             -self.integrals4[(1, -1, 1, 1)]\
                                                             +self.integrals4[(1, 1, 1, 1)])
    
        self.s17 = float(s17)
    
    def S18(self):        
        s18 = 1/16.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.co_l_2*self.co_l_3*integral4(self.l_list, [-1,-3,1,1], self.m_list)\
                                                              -self.K3*self.integrals4[(-1, -1, 1, 1)]\
                                                              +self.K3*self.integrals4[(-1, 1, 1, 1)]\
                                                              -self.co_l_2*self.co_l_3*integral4(self.l_list, [-1,3,1,1], self.m_list)\
                                                              -self.co_l_2*self.co_l_3*integral4(self.l_list, [1,-3,1,1], self.m_list)\
                                                              +self.K3*self.integrals4[(1, -1, 1, 1)]\
                                                              -self.K3*self.integrals4[(1, 1, 1, 1)]\
                                                              +self.co_l_2*self.co_l_3*integral4(self.l_list, [1,3,1,1], self.m_list))
    
        self.s18 = float(s18)
    
    def S19(self):
        s19 = -self.m**2/4.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(np.sqrt(np.pi/3)*(self.co_s_0*self.co_sprime_0*integral6(self.l_list6,[-1,-1,0,0,0,0],self.m_list6)\
                                                                                   -self.co_s_0*self.co_sprime_2*integral6(self.l_list6,[-1,-1,0,0,2,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_sprime_0*integral6(self.l_list6,[-1,-1,2,0,0,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_sprime_2*integral6(self.l_list6,[-1,-1,2,0,2,0],self.m_list6)\
                                                                                   -self.co_s_0*self.co_sprime_0*integral6(self.l_list6,[-1,1,0,0,0,0],self.m_list6)\
                                                                                   +self.co_s_0*self.co_sprime_2*integral6(self.l_list6,[-1,1,0,0,2,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_sprime_0*integral6(self.l_list6,[-1,1,2,0,0,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_sprime_2*integral6(self.l_list6,[-1,1,2,0,2,0],self.m_list6)\
                                                                                   -self.co_s_0*self.co_sprime_0*integral6(self.l_list6,[1,-1,0,0,0,0],self.m_list6)\
                                                                                   +self.co_s_0*self.co_sprime_2*integral6(self.l_list6,[1,-1,0,0,2,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_sprime_0*integral6(self.l_list6,[1,-1,2,0,0,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_sprime_2*integral6(self.l_list6,[1,-1,2,0,2,0],self.m_list6)\
                                                                                   +self.co_s_0*self.co_sprime_0*integral6(self.l_list6,[1,1,0,0,0,0],self.m_list6)\
                                                                                   -self.co_s_0*self.co_sprime_2*integral6(self.l_list6,[1,1,0,0,2,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_sprime_0*integral6(self.l_list6,[1,1,2,0,0,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_sprime_2*integral6(self.l_list6,[1,1,2,0,2,0],self.m_list6))\
                                                                                   -np.pi/(3*np.sqrt(2))*(self.K2*self.co_sprime_0*integral6(self.l_list6,[-1,-1,1,1,0,0],self.m_list6)\
                                                                                   -self.K2*self.co_sprime_2*integral6(self.l_list6,[-1,-1,1,1,2,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.co_sprime_0*integral6(self.l_list6,[-1,-1,3,1,0,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.co_sprime_2*integral6(self.l_list6,[-1,-1,3,1,2,0],self.m_list6)\
                                                                                   -self.K2*self.co_sprime_0*integral6(self.l_list6,[-1,1,1,1,0,0],self.m_list6)\
                                                                                   +self.K2*self.co_sprime_2*integral6(self.l_list6,[-1,1,1,1,2,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.co_sprime_0*integral6(self.l_list6,[-1,1,3,1,0,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.co_sprime_2*integral6(self.l_list6,[-1,1,3,1,2,0],self.m_list6)\
                                                                                   -self.K2*self.co_sprime_0*integral6(self.l_list6,[1,-1,1,1,0,0],self.m_list6)\
                                                                                   +self.K2*self.co_sprime_2*integral6(self.l_list6,[1,-1,1,1,2,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.co_sprime_0*integral6(self.l_list6,[1,-1,3,1,0,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.co_sprime_2*integral6(self.l_list6,[1,-1,3,1,2,0],self.m_list6)\
                                                                                   +self.K2*self.co_sprime_0*integral6(self.l_list6,[1,1,1,1,0,0],self.m_list6)\
                                                                                   -self.K2*self.co_sprime_2*integral6(self.l_list6,[1,1,1,1,2,0],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.co_sprime_0*integral6(self.l_list6,[1,1,3,1,0,0],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.co_sprime_2*integral6(self.l_list6,[1,1,3,1,2,0],self.m_list6)\
                                                                                   +self.co_s_0*self.K1*integral6(self.l_list6,[-1,-1,0,0,1,1],self.m_list6)\
                                                                                   -self.co_s_0*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,-1,0,0,3,1],self.m_list6)\
                                                                                   -self.co_s_2*self.K1*integral6(self.l_list6,[-1,-1,2,0,1,1],self.m_list6)\
                                                                                   +self.co_s_2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,-1,2,0,3,1],self.m_list6)\
                                                                                   -self.co_s_0*self.K1*integral6(self.l_list6,[-1,1,0,0,1,1],self.m_list6)\
                                                                                   +self.co_s_0*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,1,0,0,3,1],self.m_list6)\
                                                                                   +self.co_s_2*self.K1*integral6(self.l_list6,[-1,1,2,0,1,1],self.m_list6)\
                                                                                   -self.co_s_2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,1,2,0,3,1],self.m_list6)\
                                                                                   -self.co_s_0*self.K1*integral6(self.l_list6,[1,-1,0,0,1,1],self.m_list6)\
                                                                                   +self.co_s_0*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,-1,0,0,3,1],self.m_list6)\
                                                                                   +self.co_s_2*self.K1*integral6(self.l_list6,[1,-1,2,0,1,1],self.m_list6)\
                                                                                   -self.co_s_2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,-1,2,0,3,1],self.m_list6)\
                                                                                   +self.co_s_0*self.K1*integral6(self.l_list6,[1,1,0,0,1,1],self.m_list6)\
                                                                                   -self.co_s_0*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,1,0,0,3,1],self.m_list6)\
                                                                                   -self.co_s_2*self.K1*integral6(self.l_list6,[1,1,2,0,1,1],self.m_list6)\
                                                                                   +self.co_s_2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,1,2,0,3,1],self.m_list6))\
                                                                                   +np.pi/6*(self.K2*self.K1*integral6(self.l_list6,[-1,-1,1,1,1,1],self.m_list6)\
                                                                                   -self.K2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,-1,1,1,3,1],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.K1*integral6(self.l_list6,[-1,-1,3,1,1,1],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,-1,3,1,3,1],self.m_list6)\
                                                                                   -self.K2*self.K1*integral6(self.l_list6,[-1,1,1,1,1,1],self.m_list6)\
                                                                                   +self.K2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,1,1,1,3,1],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.K1*integral6(self.l_list6,[-1,1,3,1,1,1],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[-1,1,3,1,3,1],self.m_list6)\
                                                                                   -self.K2*self.K1*integral6(self.l_list6,[1,-1,1,1,1,1],self.m_list6)\
                                                                                   +self.K2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,-1,1,1,3,1],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.K1*integral6(self.l_list6,[1,-1,3,1,1,1],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,-1,3,1,3,1],self.m_list6)\
                                                                                   +self.K2*self.K1*integral6(self.l_list6,[1,1,1,1,1,1],self.m_list6)\
                                                                                   -self.K2*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,1,1,1,3,1],self.m_list6)\
                                                                                   -self.co_s_2*self.co_s_3*self.K1*integral6(self.l_list6,[1,1,3,1,1,1],self.m_list6)\
                                                                                   +self.co_s_2*self.co_s_3*self.co_sprime_2*self.co_sprime_3*integral6(self.l_list6,[1,1,3,1,3,1],self.m_list6)))

        self.s19 = float(s19)
    
    def S20(self):
        s20 = -1/16.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.K4*self.integrals4[(-1, -1, 1, 1)]\
                                                               -self.co_s_2*self.co_s_3*integral4(self.l_list, [-1,-1,3,1], self.m_list)\
                                                               -self.K4*self.integrals4[(-1, 1, 1, 1)]\
                                                               +self.co_s_2*self.co_s_3*integral4(self.l_list, [-1,1,3,1], self.m_list)\
                                                               -self.K4*self.integrals4[(1, -1, 1, 1)]\
                                                               +self.co_s_2*self.co_s_3*integral4(self.l_list, [1,-1,3,1], self.m_list)\
                                                               +self.K4*self.integrals4[(1, 1, 1, 1)]\
                                                               -self.co_s_2*self.co_s_3*integral4(self.l_list, [1,1,3,1], self.m_list))
    
        self.s20 = float(s20)

    def S21(self):
        s21 = 1/4.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(self.integrals4[(-1, -1, 1, 1)]\
                                                             +self.integrals4[(-1, 1, 1, 1)]\
                                                             +self.integrals4[(1, -1, 1, 1)]\
                                                             +self.integrals4[(1, 1, 1, 1)])
    
        self.s21 = float(s21)

    def S22(self):
        s22 = -self.m/8.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(np.sqrt(np.pi/3)*(self.co_s_0*self.co_sprime_0*self.integrals5[(-1, -1, 0, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(-1, -1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(-1, -1, 2, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(-1, -1, 2, 2, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(-1, 1, 0, 0, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(-1, 1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(-1, 1, 2, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(-1, 1, 2, 2, 0)]\
                                                                               +self.co_s_0*self.co_sprime_0*self.integrals5[(1,- 1, 0, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(1, -1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(1, -1, 2, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(1, -1, 2, 2, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(1, 1, 0, 0, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(1, 1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(1, 1, 2, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(1, 1, 2, 2, 0)])\
                                                             -np.sqrt(np.pi/6)*(self.co_s_0*self.K1*self.integrals5[(-1, -1, 0, 1, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(-1, -1, 2, 1, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 2, 3, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(-1, 1, 0, 1, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(-1, 1, 2, 1, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 2, 3, 1)]\
                                                                               +self.co_s_0*self.K1*self.integrals5[(1, -1, 0, 1, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(1, -1, 2, 1, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 2, 3, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(1, 1, 0, 1, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(1, 1, 2, 1, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 2, 3, 1)]))
    
        self.s22 = float(s22)
    
    def S23(self):
        # S23=-S14? or only for s=s'? Doesn't matter anymore, since involving integrals are only computed once anyway
        s23 = -self.m/8.0*self.co_lprime_0*self.co_l_0*self.co_s_0*self.co_sprime_0*(np.sqrt(np.pi/3)*(self.co_s_0*self.co_sprime_0*self.integrals5[(-1, -1, 0, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(-1, -1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(-1, -1, 2, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(-1, -1, 2, 2, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(-1, 1, 0, 0, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(-1, 1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(-1, 1, 2, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(-1, 1, 2, 2, 0)]\
                                                                               +self.co_s_0*self.co_sprime_0*self.integrals5[(1,- 1, 0, 0, 0)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.integrals5[(1, -1, 0, 2, 0)]\
                                                                               -self.co_s_2*self.co_sprime_0*self.integrals5[(1, -1, 2, 0, 0)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.integrals5[(1, -1, 2, 2, 0)]\
                                                                               -self.co_s_0*self.co_sprime_0*self.integrals5[(1, 1, 0, 0, 0)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.integrals5[(1, 1, 0, 2, 0)]\
                                                                               +self.co_s_2*self.co_sprime_0*self.integrals5[(1, 1, 2, 0, 0)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.integrals5[(1, 1, 2, 2, 0)])\
                                                             -np.sqrt(np.pi/6)*(self.co_s_0*self.K1*self.integrals5[(-1, -1, 0, 1, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(-1, -1, 2, 1, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, -1, 2, 3, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(-1, 1, 0, 1, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(-1, 1, 2, 1, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(-1, 1, 2, 3, 1)]\
                                                                               +self.co_s_0*self.K1*self.integrals5[(1, -1, 0, 1, 1)]\
                                                                               -self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 0, 3, 1)]\
                                                                               -self.co_s_2*self.K1*self.integrals5[(1, -1, 2, 1, 1)]\
                                                                               +self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, -1, 2, 3, 1)]\
                                                                               -self.co_s_0*self.K1*self.integrals5[(1, 1, 0, 1, 1)]\
                                                                               +self.co_s_0*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 0, 3, 1)]\
                                                                               +self.co_s_2*self.K1*self.integrals5[(1, 1, 2, 1, 1)]\
                                                                               -self.co_s_2*self.co_sprime_2*self.co_sprime_3*self.integrals5[(1, 1, 2, 3, 1)]))
    
        self.s23 = float(s23)


def main():
    ######################################
    # Test angular kernels
    test_angular_kernels = False
    if test_angular_kernels:
        start_time = time.time()

        # Define angular kernel parameters
        lprime, l, s, sprime, mprime, m = 10, 10, 2, 2, -4, -4

        # Compute angular kernels
        angular_kernels = AngularKernels(lprime, l, s, sprime, mprime, m)

        # Print results
        print('For [lprime,l,s,sprime,m=mprime] = [', lprime, l, s, sprime, m, ']')
        print('S1=', angular_kernels.s1)
        print('S2=', angular_kernels.s2)
        print('S3=', angular_kernels.s3)
        print('S4=', angular_kernels.s4)
        print('S5=', angular_kernels.s5)
        print('S6=', angular_kernels.s6)
        print('S7=', angular_kernels.s7)
        print('S8=', angular_kernels.s8)
        print('S9=', angular_kernels.s9)
        print('S10=', angular_kernels.s10)
        print('S11=', angular_kernels.s11)
        print('S12=', angular_kernels.s12)
        print('S13=', angular_kernels.s13)
        print('S14=', angular_kernels.s14)
        print('S15=', angular_kernels.s15)
        print('S16=', angular_kernels.s16)
        print('S17=', angular_kernels.s17)
        print('S18=', angular_kernels.s18)
        print('S19=', angular_kernels.s19)
        print('S20=', angular_kernels.s20)
        print('S21=', angular_kernels.s21)
        print('S22=', angular_kernels.s22)
        print('S23=', angular_kernels.s23)

        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")


if __name__ == '__main__':
    main()
