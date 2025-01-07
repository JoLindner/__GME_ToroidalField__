import numpy as np
import pygyre as pg
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import mesa_reader as mr
from constants import c_const as co
import os
from config import ConfigHandler


def radial_integration(r, func):
    if len(r) != len(func):
        raise ValueError('Input array sizes do not match')

    num_points  = len(r)
    integral = 0.0

    for i in range(1, num_points):
        summand = (func[i - 1] + func[i]) / 2 * (r[i] - r[i - 1])
        integral += summand

    return integral


class MesaData:
    def __init__(self, config=None):
        # Initialize configuration handler (returns single instance)
        self.config = config if config else ConfigHandler()
        # Load mesa data when initialized
        self.load_mesa_data()

    def load_mesa_data(self):
        try:
            # Load stellar model
            log_folder = self.config.get("StellarModel", "mesa_LOGS")
            DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'Stellar_model', log_folder)

            # Check if LOG folder exists
            if not os.path.exists(DATA_DIR):
                raise FileNotFoundError(
                    f"The specified log folder {log_folder} does not exist in {DATA_DIR}. Please verify the path in config.ini.")
            # Check if history.data exists
            if not os.path.exists(os.path.join(DATA_DIR, 'history.data')):
                raise FileNotFoundError(
                    f"history.data file is missing in the folder {log_folder}. Please verify the presence of the file.")

            # Load MESA profiles and history data
            LOGS = mr.MesaLogDir(DATA_DIR)
            his = mr.MesaData(os.path.join(DATA_DIR, 'history.data'))

            # Load last profile
            Star_lastprofile = LOGS.profile_data()

            # Ensure the necessary profile data attributes exist
            if not hasattr(Star_lastprofile, 'dlnRho_dr') or not hasattr(Star_lastprofile, 'logRho') or not hasattr(
                    Star_lastprofile, 'logR') or not hasattr(his, 'rsun'):
                raise AttributeError("Missing expected attributes in the profile or history data.")

            # reversed to start from 0 to R_star
            self.deriv_lnRho = Star_lastprofile.dlnRho_dr[::-1]
            self.rho_0 = np.power(10, Star_lastprofile.logRho)[::-1]  # rho_0 at outer boundary of zone in g/cm$^3$
            self.radius_array = np.power(10, Star_lastprofile.logR)[::-1]  # r/R_sun at outer boundary of zone
            # READ OUT HISTORY VARIABLES
            self.R_sun = his.rsun  # cm

            if self.radius_array is None or self.deriv_lnRho is None or self.rho_0 is None or self.R_sun is None:
                raise ValueError("MESA data is not loaded correctly.")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except AttributeError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def plot_rho_deriv(self, save=False):
        plt.figure(figsize=(11, 8))
        plt.plot(self.radius_array, self.deriv_lnRho, label='Density gradient', color='red')
        plt.xlim(self.radius_array[-1] * 0.95, self.radius_array[-1] * 1.1)
        # plt.ylim(-50, 400)
        plt.xlabel('r/R_Sun')
        plt.ylabel('d/dr ln(rho_0)')
        plt.title('Density gradient')
        plt.legend()
        plt.show()

        plt.figure(figsize=(11, 8))
        plt.plot(self.radius_array, self.rho_0, label='Density', color='red')
        plt.xlim(0, self.radius_array[-1] * 1.1)
        # plt.ylim(-50, 400)
        plt.xlabel('r/R_Sun')
        plt.ylabel('rho$_0$ in g/cm$^3$')
        plt.title('Density gradient')
        plt.legend()
        if save == True:
            output_dir = os.path.join(os.path.dirname(__file__), 'Images')
            os.makedirs(output_dir, exist_ok=True)
            DATA_DIR = os.path.join(output_dir, 'Density_gradient.png')
            plt.savefig(DATA_DIR, dpi=300, bbox_inches='tight')
        plt.show()


def magnetic_field(magnetic_field,radius_array=np.linspace(0, 1, 5000), plot_a=False ,plot_a_deriv=False, plot_B=False):
    #read in magnetic field model
    B_max=magnetic_field.B_max
    s=magnetic_field.s
    sigma=magnetic_field.sigma
    mu=magnetic_field.mu
    
    #Model Gaussian distribution
    a=1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((radius_array-mu)/sigma)**2)
    #magnetic field in e_phi direction
    Theta_Steps=2000
    theta_array_positive=np.linspace(0,np.pi/2, int(Theta_Steps/2))
    theta_array_negative=np.linspace(-np.pi/2, 0 , int(Theta_Steps/2))
    a_grid_positive, theta_positive_grid = np.meshgrid(a, theta_array_positive)
    a_grid_negative, theta_negative_grid = np.meshgrid(a, theta_array_negative)

    B_positive=-a_grid_positive*1/2*co(s,0)*(sph_harm(1,s,0,theta_positive_grid).real-sph_harm(-1,s,0, theta_positive_grid).real)
    B_negative=-a_grid_negative*1/2*co(s,0)*(sph_harm(1,s,np.pi,theta_negative_grid).real-sph_harm(-1,s,np.pi, theta_negative_grid).real)
    B_combined = np.vstack((B_negative, B_positive))

    B_scale=B_max/np.amax(B_positive)
    B_scaled=B_scale*B_combined

    a_scaled=B_scale*a

    #analytic derivatives
    a_scaled_deriv1_analytic = (-1)*B_scale/(sigma**3*np.sqrt(2*np.pi))*(radius_array-mu)*np.exp(-0.5*((radius_array-mu)/sigma)**2)
    a_scaled_deriv2_analytic = (-1)*B_scale/(sigma**3*np.sqrt(2*np.pi))*(1-(radius_array-mu)**2/sigma**2)*np.exp(-0.5*((radius_array-mu)/sigma)**2)

    if plot_a == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, a_scaled, label='Gaussian profile', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('r/R_Sun')
        plt.ylabel('a')
        plt.title('Gaussian profile for s='+str(s))
        plt.legend()

    if plot_a_deriv == True:
        fig, ax = plt.subplots(3, 1, figsize=(10, 14))
        ax[0].plot(radius_array, a_scaled,label='Model')
        ax[1].plot(radius_array, a_scaled_deriv1_analytic, ':', color='red', label='1st derivative')
        ax[2].plot(radius_array, a_scaled_deriv2_analytic, ':', color='red', label='2nd derivative')
        ax[0].set_ylabel('a in kG')
        ax[1].set_ylabel('first deriv a')
        ax[2].set_ylabel('second deriv a')

        for j in range(3):
            ax[j].legend(loc='best')
            ax[j].set_xlabel('r/R_Sun')
        ax[0].set_title('Gaussian profile for s='+str(s))
        plt.tight_layout()
        plt.show()

    
    if plot_B== True:     
        #flatten data for plotting
        theta_array=np.linspace(-np.pi/2,np.pi/2, Theta_Steps)
        r_grid, theta_grid = np.meshgrid(radius_array, theta_array)
        
        theta_flatten=theta_grid.flatten()
        r_flatten=r_grid.flatten()
        B_flatten=B_scaled.flatten()
        
        #conversion into cartesian coordinates:
        x_array=r_flatten*np.cos(theta_flatten)
        y_array=r_flatten*np.sin(theta_flatten)

        #plot
        plt.figure(figsize=(6,8))
        plt.gca().set_aspect(1, adjustable='box')
        plt.title('Magnetic field model', size=16)
        plt.xlabel('r/R$_\odot$', size=16)
        plt.ylabel('r/R$_\odot$', size=16)
        plt.xlim(0, 1.05)
        plt.ylim(-1.05,1.05)
        plt.yticks([-1.00, -0.75, -0.50, -0.25 ,0.0, 0.25,0.50,0.75, 1.00], [1.00, 0.75, 0.50, 0.25, 0.00, 0.25, 0.50, 0.75, 1.00], fontsize=14)
        plt.xticks([0.0, 0.25,0.50,0.75, 1.00], [0.00, 0.25, 0.50, 0.75, 1.00], fontsize=14)
        plt.xticks(fontsize=12)
        plt.tick_params(right=True)
        scatter=plt.scatter(x_array,y_array, c=B_flatten, s=10, cmap='seismic', alpha=0.75)
        colorbar=plt.colorbar(scatter, label='$B$ in kG', ticks=ticks_symmetric(-B_max,B_max), shrink=1)
        colorbar.set_label(label='$B$ in kG',size=16)
        colorbar.ax.tick_params(labelsize=14)
        #Half circles
        theta_half_circle = np.linspace(-np.pi/2, np.pi/2, 80)
        x_half_circle = np.cos(theta_half_circle)
        y_half_circle = np.sin(theta_half_circle)
        plt.plot(x_half_circle, y_half_circle, color='black',linewidth=0.7)
        plt.scatter(0.8*x_half_circle, 0.8*y_half_circle, color='gray',s=0.1)
        plt.scatter(0.6*x_half_circle, 0.6*y_half_circle, color='gray',s=0.1)
        plt.scatter(0.4*x_half_circle, 0.4*y_half_circle, color='gray',s=0.1)
        plt.scatter(0.2*x_half_circle, 0.2*y_half_circle, color='gray',s=0.1)
        
        # Add a polar grid
        r_ticks = np.linspace(0, 1, 30)
        theta_ticks = np.linspace(-np.pi/2, np.pi/2, 15)
        x_ticks = []
        y_ticks = []
        for r in r_ticks:
            for theta in theta_ticks:
                x_ticks.append(r * np.cos(theta))
                y_ticks.append(r * np.sin(theta))
        plt.scatter(x_ticks, y_ticks, color='gray', s=0.1)
        #plt.savefig(f'MagneticFieldA.png', dpi=300, bbox_inches='tight')
        plt.show()

    
    return a_scaled, a_scaled_deriv1_analytic, a_scaled_deriv2_analytic


def ticks_symmetric(vmin,vmax):
    #number_of_ticks=11
    ticks = np.linspace(0, vmax, 6)
    ticks_combined= np.concatenate((ticks, -ticks))
    
    return ticks_combined


def eigenfunctions(l,n,radius_array=np.linspace(0, 1, 5000),plot_eigenfunction=False,plot_deriv=False, plot_diff=False):
    try:
        # Initialize configuration handler instance
        config = ConfigHandler()

        # Read paths to detail files
        detail_path = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', config.get("StellarModel", "detail_GYRE_path"))
        name_string = "detail.l"+str(l)+".n+"+str(n)+".h5"
        DATA_DIR = os.path.join(detail_path, name_string)

        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"{DATA_DIR} not found. Check the file path and if the detail file for the eigenfunction with l={l}, n={n} exists.")

        eigenfunction = pg.read_output(DATA_DIR)
        if eigenfunction is None:
            raise ValueError(f"Unable to read eigenfunction data from {DATA_DIR}.")

        #Cubic spline interpolation
        xi_r_spline=scipy.interpolate.CubicSpline(eigenfunction['x'], eigenfunction['xi_r'].real)
        xi_r = lambda x: xi_r_spline(x)
        xi_h_spline=scipy.interpolate.CubicSpline(eigenfunction['x'], eigenfunction['xi_h'].real)
        xi_h = lambda x: xi_h_spline(x)

        if plot_deriv == True or plot_diff == True:
            # first derivative
            h = 0.000001
            xi_r_deriv1 = (xi_r(eigenfunction['x'] + h) - xi_r(eigenfunction['x'] - h)) / (2 * h)
            xi_h_deriv1 = (xi_h(eigenfunction['x'] + h) - xi_h(eigenfunction['x'] - h)) / (2 * h)
            #second derivative
            xi_r_deriv2 = (xi_r(eigenfunction['x']+2*h)-2*xi_r(eigenfunction['x'])+xi_r(eigenfunction['x']-2*h))/(4*h**2)
            xi_h_deriv2 = (xi_h(eigenfunction['x']+2*h)-2*xi_h(eigenfunction['x'])+xi_h(eigenfunction['x']-2*h))/(4*h**2)

        if plot_eigenfunction == True:

            fig, ax = plt.subplots(1, 2, figsize=(11, 8))
            ax[0].scatter(eigenfunction['x'], eigenfunction['xi_r'].real, label='xi_r/R', color='red', s=0.05)
            ax[1].scatter(eigenfunction['x'], eigenfunction['xi_h'].real, label='xi_h/R', color='blue', s=0.05)
            ax[0].set_xlabel('r/R_Sun')
            ax[0].set_ylabel('xi_r/R_Sun')
            ax[1].set_xlabel('r/R_Sun')
            ax[1].set_ylabel('xi_h/R_Sun')
            ax[0].set_title('radial displacement for n=+' + str(n) + ", l=" + str(l))
            ax[1].set_title('horizontal displacement for n=+' + str(n) + ", l=" + str(l))

            for j in range(2):
                ax[j].legend(loc='best')
            plt.tight_layout()
            plt.show()
            # plt.savefig('Bilder/Eigenfunctions_n4.png')

        if plot_deriv == True:
            x_array = np.linspace(0, np.max(eigenfunction['x']), num=10000)
            fig, ax = plt.subplots(3, 2, figsize=(10, 14))
            ax[0][0].plot(x_array, xi_r_spline(x_array), label='CubicSpline radial displacement')
            ax[0][0].scatter(eigenfunction['x'], eigenfunction['xi_r'].real, label='GYRE_data xi_r', s=0.05, color='red')
            ax[0][1].plot(x_array, xi_h_spline(x_array), label='CubicSpline horizontal displacement')
            ax[0][1].scatter(eigenfunction['x'], eigenfunction['xi_h'].real, label='GYRE_data xi_h', s=0.05, color='red')
            ax[0][0].set_title('radial displacement for n=+' + str(n) + ", l=" + str(l))
            ax[0][1].set_title('horizontal displacement for n=+' + str(n) + ", l=" + str(l))
            # ax[0].set_xlim(0.98, 1.005)
            ax[1][0].plot(x_array, xi_r_spline(x_array, nu=1), '--', label='1st deriv xi_r')
            ax[1][0].plot(eigenfunction['x'], xi_r_deriv1, ':', label='1st deriv calc xi_r')
            ax[1][0].set_xlim(0.98, 1.005)
            ax[2][0].plot(x_array, xi_r_spline(x_array, nu=2), '--', label='2nd deriv xi_r')
            ax[2][0].plot(eigenfunction['x'], xi_r_deriv2, ':', label='2nd deriv calc xi_r')
            ax[2][0].set_xlim(0.994, 1.005)
            ax[1][1].plot(x_array, xi_h_spline(x_array, nu=1), '--', label='1st deriv xi_h')
            ax[1][1].plot(eigenfunction['x'], xi_h_deriv1, ':', label='1st deriv calc xi_h')
            ax[1][1].set_xlim(0.98, 1.005)
            ax[2][1].plot(x_array, xi_h_spline(x_array, nu=2), '--', label='2nd deriv xi_h')
            ax[2][1].plot(eigenfunction['x'], xi_h_deriv2, ':', label='2nd deriv calc xi_h')
            ax[2][1].set_xlim(0.994, 1.005)

            for j in range(3):
                for i in range(2):
                    ax[j][i].legend(loc='upper left')
                    ax[j][i].set_xlabel('r/R_sun')
            plt.tight_layout()
            plt.show()

        if plot_diff == True:
            diff_spline_r = abs(eigenfunction['xi_r'].real - xi_r_spline(eigenfunction['x']))
            diff_spline_h = abs(eigenfunction['xi_h'].real - xi_h_spline(eigenfunction['x']))
            diff_deriv1_r = abs(xi_r_deriv1 - xi_r_spline(eigenfunction['x'], nu=1))
            diff_deriv1_h = abs(xi_h_deriv1 - xi_h_spline(eigenfunction['x'], nu=1))
            diff_deriv2_r = abs(xi_r_deriv2 - xi_r_spline(eigenfunction['x'], nu=2))
            diff_deriv2_h = abs(xi_h_deriv2 - xi_h_spline(eigenfunction['x'], nu=2))

            fig, ax = plt.subplots(3, 2, figsize=(10, 14))
            ax[0][0].plot(eigenfunction['x'], diff_spline_r, label='Diff Spline and radial displacement')
            ax[0][1].plot(eigenfunction['x'], diff_spline_h, label='Diff Spline and horizontal displacement')
            ax[0][0].set_title('radial displacement for n=+' + str(n) + ", l=" + str(l))
            ax[0][1].set_title('horizontal displacement for n=+' + str(n) + ", l=" + str(l))
            # ax[0].set_xlim(0.98, 1.005)
            ax[1][0].plot(eigenfunction['x'], diff_deriv1_r, ':', label='Diff 1st deriv xi_r')
            # ax[1][0].set_xlim(0.98, 1.005)
            ax[1][1].plot(eigenfunction['x'], diff_deriv1_h, ':', label='Diff 1st deriv xi_h')
            # ax[2][0].set_xlim(0.994, 1.005)
            ax[2][0].plot(eigenfunction['x'], diff_deriv2_r, ':', label='Diff 2nd deriv  xi_r')
            # ax[1][1].set_xlim(0.98, 1.005)
            ax[2][1].plot(eigenfunction['x'], diff_deriv2_h, ':', label='Diff 2nd deriv xi_h')
            # ax[2][1].set_xlim(0.994, 1.005)

            for j in range(3):
                for i in range(2):
                   ax[j][i].legend(loc='upper left')
                   ax[j][i].set_xlabel('r/R_sun')
            plt.tight_layout()
            plt.show()

        return xi_r(radius_array), xi_h(radius_array), xi_r_spline(radius_array, nu=1), xi_h_spline(radius_array,nu=1), xi_r_spline(radius_array, nu=2), xi_h_spline(radius_array, nu=2)

    except FileNotFoundError as e:
        print(f"File error: {e}")
        raise

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


#Radial Kernels
def R1(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None, deriv_lnRho=None, plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array*deriv_lnRho\
        +magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]\
        *deriv_lnRho*magnetic_field(magnetic_field_s,radius_array)[1]\
        +magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array**2\
        -magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]\
        /radius_array*magnetic_field(magnetic_field_s,radius_array)[1]\
        -magnetic_field(magnetic_field_s,radius_array)[0]*eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]\
        /radius_array*magnetic_field(magnetic_field_sprime,radius_array)[1]\
        -eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]*magnetic_field(magnetic_field_s,radius_array)[1]\
        *magnetic_field(magnetic_field_sprime,radius_array)[1]\
        -2*magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]\
        /radius_array*eigenfunctions(l,n,radius_array)[2]\
        -magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[4]\
        -2*magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]*eigenfunctions(l,n,radius_array)[2]\
        *magnetic_field(magnetic_field_s,radius_array)[1]\
        -magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]\
        *magnetic_field(magnetic_field_s,radius_array)[2]  #kG^2        
        
        
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_1$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_1$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()
        
    return radial_integration(radius_array, func*radius_array**2), func

def R2(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None,plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[1]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array**2\
        +magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(l,n,radius_array)[1]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array\
        *magnetic_field(magnetic_field_s,radius_array)[1]
        
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_2$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_2$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()

    return radial_integration(radius_array, func*radius_array**2), func

def R3(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None,plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_sprime,radius_array)[0]*magnetic_field(magnetic_field_s,radius_array)[1]\
        *eigenfunctions(l,n,radius_array)[1]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array\
        +magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(l,n,radius_array)[3]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array\
        *magnetic_field(magnetic_field_s,radius_array)[0]
        
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_3$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_3$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()

    return radial_integration(radius_array, func*radius_array**2), func

def R4(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None,plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[1]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array**2\
        -magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array**2\
        +magnetic_field(magnetic_field_sprime,radius_array)[0]*eigenfunctions(l,n,radius_array)[1]*eigenfunctions(lprime,nprime,radius_array)[0]/radius_array\
        *magnetic_field(magnetic_field_s,radius_array)[1]
        
        
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_4$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_4$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()

    return radial_integration(radius_array, func*radius_array**2), func

def R5(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None,deriv_lnRho=None,plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array*deriv_lnRho\
        -magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[1]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array\
        -magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[2]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array\
        -magnetic_field(magnetic_field_s,radius_array)[1]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array
        
        
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_5$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_5$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()

    return radial_integration(radius_array, func*radius_array**2), func

def R6(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None,plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[1]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array**2 
    
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_6$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_6$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()

    return radial_integration(radius_array, func*radius_array**2), func

def R7(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None,plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array**2\
        +magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[2]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array\
        +magnetic_field(magnetic_field_s,radius_array)[1]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array
    
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_7$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_7$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()

    return radial_integration(radius_array, func*radius_array**2), func

def R8(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None,plot_kernel=False):
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
        
    func=magnetic_field(magnetic_field_s,radius_array)[0]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array**2\
        +magnetic_field(magnetic_field_s,radius_array)[1]*magnetic_field(magnetic_field_sprime,radius_array)[0]\
        *eigenfunctions(l,n,radius_array)[0]*eigenfunctions(lprime,nprime,radius_array)[1]/radius_array
    
    if plot_kernel == True:
        plt.figure(figsize=(11,8))
        plt.plot(radius_array, func, label='radial kernel $R_8$', color='red')
        #plt.xlim(0.4, 1.005)
        #plt.ylim(-50, 400)
        plt.xlabel('$r/R_\odot$')
        plt.ylabel('$R_8$ in kG$^2$')
        title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
            l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
        plt.title(title)
        plt.legend()
        plt.show()

    return radial_integration(radius_array, func*radius_array**2), func

def plot_all_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None):
    #Note: magnetic_field_sprime is not used here and is assumed to be None
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s
    
    #Logarithmic plot in the relevant regime 
    plt.figure(figsize=(11,8))
    plt.plot(radius_array, R1(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R1')
    plt.plot(radius_array, R2(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R2')
    plt.plot(radius_array, R3(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R3')
    plt.plot(radius_array, R4(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R4')
    plt.plot(radius_array, R5(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R5')
    plt.plot(radius_array, R6(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R6')
    plt.plot(radius_array, R7(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R7')
    plt.plot(radius_array, R8(l,n,lprime,nprime,radius_array,magnetic_field_s)[1], label='radial kernel R8')
    plt.xlim(0.5, 0.9)
    #plt.ylim(-50, 400)
    plt.xlabel('r/R_Sun')
    plt.ylabel('Radial Kernels [kG$^2$]')
    title = 'For [l,n,lprime,nprime]=[{},{},{},{}] and a magnetic_field with [B_max,mu,sigma,s]=[{},{},{},{}]'.format(
        l, n, lprime, nprime, magnetic_field_s.B_max, magnetic_field_s.mu, magnetic_field_s.sigma, magnetic_field_s.s)        
    plt.title(title)
    plt.legend()
    plt.yscale('symlog')
    plt.show()

def plot_several_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None):
    #Note: magnetic_field_sprime is not used here and is assumed to be None
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s

    if len(l)>10:
        print('Can only plot up to 10 different radial kernels')
        return

    colors=['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown', 'black', 'magenta', 'yellow']
    fig, ax = plt.subplots(4, 2, figsize=(10, 14), sharex=True)
    for i in range(len(l)):
        ax[0][0].plot(radius_array, R1(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1], label=f'$l={l[i]}$, $n={n[i]}$, $l^\prime={lprime[i]}$, $n^\prime={nprime[i]}$', color=colors[i])
        ax[0][1].plot(radius_array, R2(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[1][0].plot(radius_array, R3(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[1][1].plot(radius_array, R4(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[2][0].plot(radius_array, R5(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[2][1].plot(radius_array, R6(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[3][0].plot(radius_array, R7(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[3][1].plot(radius_array, R8(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])

    ax[0][0].set_ylabel('$R_1$ in kG$^2$', fontsize=14)
    ax[0][1].set_ylabel('$R_2$ in kG$^2$', fontsize=14)
    ax[1][0].set_ylabel('$R_3$ in kG$^2$', fontsize=14)
    ax[1][1].set_ylabel('$R_4$ in kG$^2$', fontsize=14)
    ax[2][0].set_ylabel('$R_5$ in kG$^2$', fontsize=14)
    ax[2][1].set_ylabel('$R_6$ in kG$^2$', fontsize=14)
    ax[3][0].set_ylabel('$R_7$ in kG$^2$', fontsize=14)
    ax[3][1].set_ylabel('$R_8$ in kG$^2$', fontsize=14)
    for j in range(4):
        for i in range(2):
            ax[j][i].set_xlim(0.563, 0.9)
            #ax[j][i].legend(loc='upper left', fontsize=14)
            #ax[j][i].set_yscale('symlog', linthresh=0.01)
            ax[3][i].set_xlabel('r/R$_{\odot}$', fontsize=14)
            ax[j][i].tick_params(axis='both', which='major', labelsize=14)
    #ax[3][1].legend(loc='lower left', fontsize=12)
    ncol=min(len(l),3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), fontsize=14, ncol=ncol)
    plt.tight_layout()
    plt.savefig(f'Images/RadialKernels_l{l}_n{n}_lprime{lprime}_nprime{nprime}.pdf', dpi=300, bbox_inches='tight')
    #plt.show()

def plot_radialKernels_for_Defense(l,n,lprime,nprime,radius_array,magnetic_field_s,magnetic_field_sprime=None):
    #Note: magnetic_field_sprime is not used here and is assumed to be None
    if magnetic_field_sprime is None:
        magnetic_field_sprime = magnetic_field_s

    if len(l)>10:
        print('Can only plot up to 10 different radial kernels')
        return

    colors=['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown', 'black', 'magenta', 'yellow']
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    for i in range(len(l)):
        ax[0][0].plot(radius_array, R1(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1], label=f'$l={l[i]}$, $n={n[i]}$, $l^\prime={lprime[i]}$, $n^\prime={nprime[i]}$', color=colors[i])
        ax[0][1].plot(radius_array, R2(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[1][0].plot(radius_array, R3(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])
        ax[1][1].plot(radius_array, R4(l[i],n[i],lprime[i],nprime[i],radius_array,magnetic_field_s)[1],
                      color=colors[i])

    ax[0][0].set_ylabel('$R_1$ in kG$^2$', fontsize=14)
    ax[0][1].set_ylabel('$R_2$ in kG$^2$', fontsize=14)
    ax[1][0].set_ylabel('$R_3$ in kG$^2$', fontsize=14)
    ax[1][1].set_ylabel('$R_4$ in kG$^2$', fontsize=14)
    for j in range(2):
        for i in range(2):
            ax[j][i].set_xlim(0.563, 0.9)
            #ax[j][i].legend(loc='upper left', fontsize=14)
            #ax[j][i].set_yscale('symlog', linthresh=0.01)
            ax[1][i].set_xlabel('r/R$_{\odot}$', fontsize=14)
            ax[j][i].tick_params(axis='both', which='major', labelsize=14)
    #ax[3][1].legend(loc='lower left', fontsize=12)
    ncol=min(len(l),3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), fontsize=14, ncol=ncol)
    plt.tight_layout()
    plt.savefig(f'Images/RadialKernels_l{l}_n{n}_lprime{lprime}_nprime{nprime}_firstpart.pdf', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    for i in range(len(l)):
        ax[0][0].plot(radius_array, R5(l[i], n[i], lprime[i], nprime[i], radius_array, magnetic_field_s)[1], label=f'$l={l[i]}$, $n={n[i]}$, $l^\prime={lprime[i]}$, $n^\prime={nprime[i]}$',
                      color=colors[i])
        ax[0][1].plot(radius_array, R6(l[i], n[i], lprime[i], nprime[i], radius_array, magnetic_field_s)[1],
                      color=colors[i])
        ax[1][0].plot(radius_array, R7(l[i], n[i], lprime[i], nprime[i], radius_array, magnetic_field_s)[1],
                      color=colors[i])
        ax[1][1].plot(radius_array, R8(l[i], n[i], lprime[i], nprime[i], radius_array, magnetic_field_s)[1],
                      color=colors[i])

    ax[0][0].set_ylabel('$R_5$ in kG$^2$', fontsize=14)
    ax[0][1].set_ylabel('$R_6$ in kG$^2$', fontsize=14)
    ax[1][0].set_ylabel('$R_7$ in kG$^2$', fontsize=14)
    ax[1][1].set_ylabel('$R_8$ in kG$^2$', fontsize=14)
    for j in range(2):
        for i in range(2):
            ax[j][i].set_xlim(0.563, 0.9)
            # ax[j][i].legend(loc='upper left', fontsize=14)
            # ax[j][i].set_yscale('symlog', linthresh=0.01)
            ax[1][i].set_xlabel('r/R$_{\odot}$', fontsize=14)
            ax[j][i].tick_params(axis='both', which='major', labelsize=14)
    # ax[3][1].legend(loc='lower left', fontsize=12)
    ncol = min(len(l), 3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), fontsize=14, ncol=ncol)
    plt.tight_layout()
    plt.savefig(f'Images/RadialKernels_l{l}_n{n}_lprime{lprime}_nprime{nprime}_secondpart.pdf', dpi=300, bbox_inches='tight')
    #plt.show()


class MagneticField:
    def __init__(self, B_max, mu, sigma, s):
        self.B_max = B_max  # in kG
        self.mu = mu  # in R_sun
        self.sigma = sigma  # in R_sun
        self.s = s  # harmonic degree

def main():
    # Initialize configuration handler (first instance)
    config = ConfigHandler("config.ini")

    # Initialize magnetic field:
    B_max = config.getfloat("MagneticFieldModel", "B_max")
    mu = config.getfloat("MagneticFieldModel", "mu")
    sigma = config.getfloat("MagneticFieldModel", "sigma")
    s = config.getint("MagneticFieldModel", "s")

    # Initialize magnetic field model
    magnetic_field_s = MagneticField(B_max=B_max, mu=mu, sigma=sigma, s=s)

    # Initialize stellar model (MESA data)
    mesa_data = MesaData(config=config)

    #plot mesa_data
    #mesa_data.plot_rho_deriv(save=False)

    # Extract radius_array from MESA class
    radius_array = mesa_data.radius_array

    #plot magnetic field
    magnetic_field(magnetic_field_s, radius_array, False, False, False)

    #TEST AREA:

    '''
    #Plot radial kernel of l=5 n=0
    l=[5,5,5,5,5]
    n=[12,12,12,12,12]
    lprime=[5,38,94,141,237]
    nprime=[12,6,3,2,1]
    plot_radialKernels_for_Defense(l,n,lprime,nprime,radius_array,magnetic_field_s)
    #plot_several_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s)
    '''
    '''
    l=[140]
    n=[10]
    lprime=[140]
    nprime=[10]
    plot_several_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s)
    '''
    '''
    l=[5,5,5,5]
    n=[18,18,18,18]
    lprime=[5,29,42,225]
    nprime=[18,12,10,3]
    #plot_several_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s)
    plot_radialKernels_for_Defense(l,n,lprime,nprime,radius_array,magnetic_field_s)
    '''
    #Plot radial kernels of l=5 n=0 eigenspace
    '''
    lprime=[5,5,5,5,5]
    nprime=[0,0,0,0,0]
    l=[3,4,5,6,7]
    n=[1,0,0,0,0]
    plot_several_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s)
     '''
    '''
    #Plot radial kernels of l=5 n=6 eigenspace
    l=[5,5,5,5,5,5,5]
    n=[6,6,6,6,6,6,6]
    lprime=[2,5,9,23,66,154,155]
    nprime=[7,6,5,3,1,0,0]
    plot_radialKernels_for_Defense(l,n,lprime,nprime,radius_array,magnetic_field_s)
    '''
    '''
    lprime=[5,5,5]
    nprime=[6,6,6]
    l=[66,154,155]
    n=[1,0,0]
    plot_several_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s)
    '''
    #eigenfunctions(l,n, radius_array, False, False, plot_diff = False)
    #eigenfunctions(lprime,nprime, radius_array, False, False)
    #MESA_structural_data(False)

    '''
    plot_all_radialKernels(l,n,lprime,nprime,radius_array,magnetic_field_s)

    print('For [l,n,lprime,nprime]=['+str(l),str(n),str(lprime),str(nprime)+'] and a magnetic_field with [B_max,mu,sigma,s]=['\
          ,str(magnetic_field_s.B_max),str(magnetic_field_s.mu),str(magnetic_field_s.sigma),str(magnetic_field_s.s)+']\n')

    print('R1='+str(R1(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    print('R2='+str(R2(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    print('R3='+str(R3(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    print('R4='+str(R4(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    print('R5='+str(R5(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    print('R6='+str(R6(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    print('R7='+str(R7(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    print('R8='+str(R8(l,n,lprime,nprime,radius_array,magnetic_field_s,plot_kernel=True)[0])+' kG^2*R_sun^3')
    '''

if __name__== '__main__':
    main()
    
