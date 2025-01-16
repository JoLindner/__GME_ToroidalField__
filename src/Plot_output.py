import numpy as np
import os
import GeneralMatrixElements_Parallel as GMEP
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.colors import ListedColormap
from config import ConfigHandler
import pygyre as pg
from scipy import optimize
import scipy.interpolate
import radial_kernels


def plot_frequency_shifts(l, n, plot_relative_shifts=True, plot_absolute_shifts=True, plot_combined=False, path=None, save=False):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Frequency Shift File does not exist: {path}")

    data = np.loadtxt(path, comments='#', delimiter=' ')
    freq_shift = data[:,0] #nHz
    m = data[:,3]
    freq = GMEP.frequencies_GYRE(l,n)*10**3   #nHz
    #relative_shift=(freq+freq_shift)/freq-1
    relative_shift = freq_shift/freq
    if plot_relative_shifts == True:
        plt.figure(figsize=(8, 7))
        plt.scatter(m/l,relative_shift, marker='x')
        plt.axhline(freq/freq-1, linestyle='--', label='unperturbed frequency', alpha=0.5)
        plt.xlabel('$m/l$', fontsize=14)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.xlim([-1.05,1.05])
        data_range = np.max(relative_shift) - np.min(relative_shift)
        margin = 0.05 * data_range
        plt.ylim([min(freq/freq-1 - 3*margin,np.min(relative_shift) - margin), np.max(relative_shift) + margin])
        plt.ylabel(r'$\frac{\delta\nu}{\nu_0}$', fontsize=14)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().yaxis.get_offset_text().set_position((-0.05, 1))
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.title(f'Relative frequency shifts of multiplet $l={l},n={n}$', fontsize=16)
        if save == True:
            output_dir = os.path.join(os.path.dirname(__file__), 'Images')
            os.makedirs(output_dir, exist_ok=True)
            DATA_DIR = os.path.join(output_dir, f'Relative_frequency_shifts_l{l}_n{n}.png')
            plt.savefig(DATA_DIR, dpi=300, bbox_inches='tight')
        plt.show()

    if plot_absolute_shifts == True:
        #Plot frequency shift diagram
        plot_array = np.zeros((len(m),4), dtype=float)
        plot_array[:, :2] = freq
        for i in range(0,len(m)):
            plot_array[i, 2:4] = freq+freq_shift[i]
        plt.figure(figsize=(8, 7))
        plot_numbering = [1,2,2.3,3.3]
        shift_dict = {}
        for i, shift in enumerate(freq_shift):
            if shift not in shift_dict:
                shift_dict[shift] = []
            shift_dict[shift].append(m[i])
        plotted_shifts = set()
        for index, row in enumerate(plot_array):
            shift = freq_shift[index]
            if shift not in plotted_shifts:
                plotted_shifts.add(shift)
                label = ', '.join([f'm={int(m_val)}' for m_val in shift_dict[shift]])
                plt.plot(plot_numbering, row - freq, label=label)
        plt.gca().xaxis.set_visible(False)
        plt.ylabel(r'$\delta\nu=\nu-\nu_0$ in nHz', fontsize=14)
        plt.title(f'Frequency shifts of multiplet $l={l},n={n}$', fontsize=16)
        plt.legend(loc='upper left', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(axis='y', alpha=0.5)
        if save == True:
            output_dir = os.path.join(os.path.dirname(__file__), 'Images')
            os.makedirs(output_dir, exist_ok=True)
            DATA_DIR = os.path.join(output_dir, f'Absolute_frequency_shifts_l{l}_n{n}.png')
            plt.savefig(DATA_DIR, dpi=300, bbox_inches='tight')
        plt.show()

    if plot_combined==True:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].scatter(m/l,relative_shift, marker='x')
        ax[0].axhline(freq/freq-1, linestyle='--', label='unperturbed frequency', alpha=0.5)
        ax[0].set_xlabel('$m/l$', fontsize=14)
        ax[0].set_xticks([-1,-0.5,0,0.5,1])
        ax[0].set_xlim([-1.05,1.05])
        data_range = np.max(relative_shift) - np.min(relative_shift)
        margin = 0.05 * data_range
        ax[0].set_ylim([min(freq/freq-1 - 3*margin,np.min(relative_shift) - margin), np.max(relative_shift) + margin])
        ax[0].set_ylabel(r'$\frac{\delta\nu}{\nu_0}$', fontsize=14)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().yaxis.get_offset_text().set_position((-0.05, 1))
        ax[0].tick_params(axis='both', which='major', labelsize=12)

        #ax[0].set_title(f'Relative frequency shifts of multiplet $l={l},n={n}$', fontsize=16)
        #plt.legend()
        plot_array = np.zeros((len(m),4), dtype=float)
        plot_array[:, :2] = freq
        for i in range(0,len(m)):
            plot_array[i, 2:4] = freq+freq_shift[i]
        plot_numbering = [1,2,2.3,3.3]
        shift_dict = {}
        for i, shift in enumerate(freq_shift):
            if shift not in shift_dict:
                shift_dict[shift] = []
            shift_dict[shift].append(m[i])
        plotted_shifts = set()
        for index, row in enumerate(plot_array):
            shift = freq_shift[index]
            if shift not in plotted_shifts:
                plotted_shifts.add(shift)
                label = ', '.join([f'm={int(m_val)}' for m_val in shift_dict[shift]])
                ax[1].plot(plot_numbering, row - freq, label=label)
        plt.gca().xaxis.set_visible(False)
        ax[1].set_ylabel(r'$\delta\nu=\nu-\nu_0$ in nHz', fontsize=14)
        #ax[1].set_title(f'Frequency shifts of multiplet $l={l},n={n}$', fontsize=16)
        ax[1].legend(loc='upper left', fontsize=12)
        ax[1].tick_params(axis='both', which='major', labelsize=12)
        ax[1].grid(axis='y', alpha=0.5)
        fig.suptitle(f'l={l}, n={n}', fontsize=16)
        plt.tight_layout()
        if save == True:
            output_dir = os.path.join(os.path.dirname(__file__), 'Images')
            os.makedirs(output_dir, exist_ok=True)
            DATA_DIR = os.path.join(output_dir, f'Frequency_shifts_combined_l{l}_n{n}.pdf')
            plt.savefig(DATA_DIR, dpi=300, bbox_inches='tight')
        plt.show()


def plot_frequency_shifts_overlaid(l_list, n_list, path, eigentag, save=False):
    name_string_list = []
    for i in range(len(l_list)):
        if eigentag is None or eigentag == 'Full':
            name_string_list.append(f'freq_shifts_{l_list[i]}_{n_list[i]}_full.txt')
        elif eigentag == 'FirstApprox':
            name_string_list.append(f'freq_shifts_{l_list[i]}_{n_list[i]}_first_approx.txt')
        elif eigentag == 'SelfCoupling':
            name_string_list.append(f'freq_shifts_{l_list[i]}_{n_list[i]}_self_coupling.txt')
        else:
            # Raise an error if the eigen_tag is invalid
            raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')

        file_path = os.path.join(path, name_string_list[i])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Frequency Shift File does not exist: {file_path}")

    data_list = []
    freq_shift_list = []
    m_list = []
    freq_list = []
    relative_shift_list = []

    plt.figure(figsize=(11, 8))
    for i in range(len(l_list)):
        data_list.append(np.loadtxt(os.path.join(path, name_string_list[i]), comments='#', delimiter=' '))
        freq_shift_list.append(data_list[i][:,0])  # nHz
        m_list.append(data_list[i][:,3])
        freq_list.append(GMEP.frequencies_GYRE(l_list[i],n_list[i])*10**3)  # nHz
        relative_shift_list.append(freq_shift_list[i]/freq_list[i])
        plt.scatter(m_list[i]/l_list[i],relative_shift_list[i], label=f'l={l_list[i]}, n={n_list[i]}')
    plt.axhline(freq_list[0]/freq_list[0]-1, linestyle='--', label='unperturbed frequency', alpha=0.5)
    plt.xlabel('$m/l$', fontsize=14)
    plt.xticks([-1,-0.5,0,0.5,1])
    plt.xlim([-1.05,1.05])
    #data_range = np.max(relative_shift_list[-1]) - np.min(relative_shift_list[-1])
    #margin = 0.05 * data_range
    #plt.ylim([min((-3)*margin,np.min(relative_shift_list[0]) - margin), np.max(relative_shift_list[-1]) + margin])
    plt.ylabel(r'$\frac{\delta\nu}{\nu_0}$', fontsize=14)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().yaxis.get_offset_text().set_position((-0.05, 1))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title(r'Comparison of Relative Frequency Shifts', fontsize=16)
    ncol = min(len(l_list)+1, 5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=14, ncol=ncol)
    plt.tight_layout()
    if save == True:
        output_dir = os.path.join(os.path.dirname(__file__), 'Images')
        os.makedirs(output_dir, exist_ok=True)
        DATA_DIR = os.path.join(output_dir, f'Relative_frequency_shifts_overlaid_l={l_list}_n={n_list}.png')
        plt.savefig(DATA_DIR, dpi=300, bbox_inches='tight')
    plt.show()


def plot_output_summary(l_list, n_list, path, eigentag, lower_tp=None, r_thresh=None, save=False):
    # Load lower turning points
    lower_turning_points_dict = {(int(l), int(n)): r_ltp for l, n, r_ltp in lower_tp} if lower_tp is not None else {}

    # Load data
    frequency_shifts = []
    averaged_shifts = []
    for i in range(len(l_list)):
        if eigentag is None or eigentag == 'Full':
            name_string = f'freq_shifts_{l_list[i]}_{n_list[i]}_full.txt'
        elif eigentag == 'FirstApprox':
            name_string = f'freq_shifts_{l_list[i]}_{n_list[i]}_first_approx.txt'
        elif eigentag == 'SelfCoupling':
            name_string = f'freq_shifts_{l_list[i]}_{n_list[i]}_self_coupling.txt'
        else:
            # Raise an error if the eigen_tag is invalid
            raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')
        file_path = os.path.join(path, name_string)

        try:
            frequency_shifts.append(np.loadtxt(file_path, comments='#', delimiter=' '))

            rtp_value = lower_turning_points_dict.get((int(l_list[i]), int(n_list[i])), None)

            averaged_shifts_dict = {
                'l': l_list[i],
                'n': n_list[i],
                'freq': GMEP.frequencies_GYRE(l_list[i], n_list[i]),
                'av_shift': np.mean(frequency_shifts[i][:, 0]),
                'rtp': rtp_value
            }
            averaged_shifts.append(averaged_shifts_dict)

        except FileNotFoundError:
            #print(f"Error: Unable to load {file_path}")
            continue

    freq_values = [entry['freq'] for entry in averaged_shifts]
    av_shifts = [entry['av_shift'] for entry in averaged_shifts]
    l_values = [entry['l'] for entry in averaged_shifts]
    if lower_tp is not None:
        rtp_values = [entry['rtp'] for entry in averaged_shifts]

    # plot data
    plt.figure(figsize=(12, 8))
    plt.xlabel('r$_t/R_\odot$' if lower_tp is not None else '$\\nu$ in µHz', size=16)
    plt.ylabel(r'$\delta\nu$ in nHz', size=16)
    scatter = plt.scatter(rtp_values if lower_tp is not None else freq_values, av_shifts, c=l_values, s=5, cmap='gnuplot', alpha=0.75)
    colorbar = plt.colorbar(scatter, label='$l$')
    colorbar.set_label(label='harmonic degree $l$', size=16)
    ticks = colorbar.get_ticks()
    min_l, max_l = np.min(l_values), np.max(l_values)
    ticks = [tick for tick in ticks if min_l <= tick <= max_l]
    ticks = sorted(set(ticks + [min_l, max_l]))
    colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=14)

    unique_l = np.unique(l_values)
    colors = scatter.cmap(scatter.norm(unique_l))
    for l_value, color in zip(unique_l, colors):
        subset_indices = [i for i, val in enumerate(l_values) if val == l_value]
        plt.plot(np.array(rtp_values if lower_tp is not None else freq_values)[subset_indices], np.array(av_shifts)[subset_indices], color=color, alpha=0.5, linewidth=1)

    if lower_tp is not None and r_thresh is not None:
        plt.axvline(r_thresh, color='black', linestyle='--', label='R$_{thresh}$'+f' = {r_thresh}')

    plt.ylim(min(0, min(av_shifts)), max(av_shifts) * 1.05)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    if save:
        output_dir = os.path.join(os.path.dirname(__file__), 'Images')
        os.makedirs(output_dir, exist_ok=True)
        if lower_tp is not None:
            DATA_DIR = os.path.join(output_dir, f'Freq_shift_summary_lower_tp.pdf')
        else:
            DATA_DIR = os.path.join(output_dir, f'Freq_shift_summary.pdf')
        plt.savefig(DATA_DIR, dpi=300, bbox_inches='tight')
    plt.show()


def lower_turning_points(path_to_summary_file, mesa_data):
    # Read in summary file
    summary_file = pg.read_output(path_to_summary_file)
    l_group = summary_file.group_by('l')

    # Cubic Spline of sound speed
    c_sound_spline = scipy.interpolate.CubicSpline(mesa_data.radius_array, mesa_data.c_sound,
                                                   bc_type='natural')
    c_sound_spline_func = lambda x: c_sound_spline(x)

    # R_sun
    R_sun = mesa_data.R_sun

    # Lower turning point computation
    results = []
    i = 0
    # print(optimize.root_scalar(lower_turning_point_fixed_point_func,method='secant', x0=0.5, bracket=[0,1], args=(l,n)))
    for l in l_group['l']:
        if l != 0:  # omit radial oscillations
            n_pg = l_group[i]['n_pg']
            if n_pg != 0:  # omit f modes
                r_ltp = optimize.root_scalar(lower_turning_point_fixed_point_func, method='toms748', x0=0.5,
                                             bracket=[0.001, 1], args=(i, l, l_group, R_sun, c_sound_spline_func))
                results.append((int(l), int(n_pg), r_ltp.root))
        i += 1
    lower_tp = np.array(results, dtype=object)

    return lower_tp


def lower_turning_point_fixed_point_func(x, index, l, l_group, R_sun, c_sound_spline_func):
    omega = 2*np.pi*l_group[index]['freq'].real*10**(-6)   # in Hz
    return omega**2/(l*(l+1))-(c_sound_spline_func(x))**2/(x*R_sun)**2


# SUPERMATRIX PLOTS
def plot_supermatrix_var(l, n, linthresh):
    # load supermatrix
    name_string = f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')
    supermatrix_array = supermatrix_array.astype(float)

    # Load index_map
    name_string = f'index_map_supermatrix_array_{l}_{n}.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    index_map = load_index_map_from_file(data_dir)

    # linthresh = np.min(np.abs(supermatrix_array[supermatrix_array != 0]))
    vmax = np.max(np.abs(supermatrix_array))
    norm = SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, linscale=2)
    fig, ax = plt.subplots(figsize=(10, 9))
    cax = ax.pcolormesh(supermatrix_array, norm=norm, cmap='seismic', alpha=1)
    # Add colorbar
    cb = plt.colorbar(cax, ax=ax, label=f'Supermatrixelement in $\mu$Hz$^2$', pad=0.02)
    cb.set_label(label=f'Supermatrixelement in $\mu$Hz$^2$', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    # plt.xlabel('l, n')
    # plt.ylabel('lprime, nprime')

    major_ticks = []
    minor_ticks = []
    tickslabel = []
    minor_tickslabel = []
    for i in range(len(index_map[0])):
        column = index_map[0][i]
        minor_ticks.append(i + 0.5)
        minor_tickslabel.append(f'm={column["m"]}')
        l = column['l']
        if l == abs(column['m']):
            major_ticks.append(i + 0.5)
            tickslabel.append(f'l={l}, n={column["n"]}, m={column["m"]}')

    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    for minor_tick in ax.xaxis.get_minorticklocs():
        ax.text(minor_tick, ax.get_ylim()[0] - 1, minor_tickslabel[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='top', rotation=90)
    for minor_tick in ax.yaxis.get_minorticklocs():
        ax.text(ax.get_xlim()[0] - 1, minor_tick, minor_tickslabel[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='top')

    ax.set_xticklabels(tickslabel, rotation=90, fontsize=12)
    ax.set_yticklabels(tickslabel, fontsize=12)

    plt.title(f'Supermatrix for multiplet around l={l}, n={n}', fontsize=16)
    plt.savefig(f'Images/Supermatrix_Pixel_Plot_l={l}_n={n}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_supermatrix(l, n, linthresh, trunc=None):
    # load supermatrix
    name_string = f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')
    supermatrix_array = supermatrix_array.astype(float)

    # Load index_map
    name_string = f'index_map_supermatrix_array_{l}_{n}.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    index_map = load_index_map_from_file(data_dir)

    if trunc is not None:
        index_map_l = np.unique(index_map['l'])[:-trunc]
        mask = np.isin(index_map['l'], index_map_l) & np.isin(index_map['lprime'], index_map_l)
        true_indices = np.argwhere(mask[0] == False)
        index_map = np.delete(index_map, true_indices, axis=0)
        index_map = np.delete(index_map, true_indices, axis=1)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=0)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=1)

    # print elements above threshold:
    thresh = linthresh
    data_above = []
    for i in range(len(supermatrix_array)):
        for j in range(len(supermatrix_array[i])):
            if np.abs(supermatrix_array[i][j]) > thresh:
                data_above.append((supermatrix_array[i][j], index_map[i][j]))
    sorted_data_above = sorted(data_above, key=lambda x: x[1][0])
    for value, index in sorted_data_above:
        print(value, index)

    # linthresh = np.min(np.abs(supermatrix_array[supermatrix_array != 0]))
    # print(linthresh)
    vmax = np.max(np.abs(supermatrix_array))
    norm = SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, linscale=2)
    fig, ax = plt.subplots(figsize=(16, 16))
    cax = ax.pcolormesh(supermatrix_array, norm=norm, cmap='seismic', alpha=1)
    # Add colorbar
    cb = plt.colorbar(cax, ax=ax, label=f'Supermatrixelement in µHz$^2$', pad=0.02, shrink=0.84)
    cb.set_label(label=f'Supermatrixelement in µHz$^2$', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    plt.title(f'Supermatrix for quasi-degenerate multiplet around l={l}, n={n}', fontsize=16)

    major_ticks = []
    minor_ticks = []
    tickslabel_x = []
    tickslabel_y = []
    minor_tickslabel_x = []
    minor_tickslabel_y = []
    for i in range(len(index_map[0])):
        column = index_map[0][i]
        l_co = column['l']
        # tick only every x-th m and m=0
        every_x_tick = 2
        if abs(column['m']) % every_x_tick == 0:
            minor_ticks.append(i + 0.5)
            minor_tickslabel_x.append(f'm={column["m"]}')
            minor_tickslabel_y.append(f'm\'={column["m"]}')

        if l_co == -column['m']:
            major_ticks.append(i)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')
        elif i == len(index_map[0]) - 1:
            major_ticks.append(i + 1)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')

    for i in range(len(major_ticks) - 1):
        x_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        # displaces first label
        if i == 0 and trunc is not None:
            ax.text(x_mid, ax.get_ylim()[0] - 0.090 * (ax.get_ylim()[1] - ax.get_ylim()[0]), tickslabel_x[i],
                    ha='center', va='center', fontsize=16)  # adjust factor for y position
        else:
            ax.text(x_mid, ax.get_ylim()[0] - 0.065 * (ax.get_ylim()[1] - ax.get_ylim()[0]), tickslabel_x[i],
                    ha='center', va='center', fontsize=16)  # adjust factor for y position

    for i in range(len(major_ticks) - 1):
        y_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        if i == 0 and trunc is not None:
            ax.text(ax.get_xlim()[0] - 0.090 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_mid, tickslabel_y[i],
                    ha='center', va='center', fontsize=16, rotation=90)  # adjust factor for y position
        else:
            ax.text(ax.get_xlim()[0] - 0.065 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_mid, tickslabel_y[i],
                    ha='center', va='center', fontsize=16, rotation=90)  # adjust factor for y position

    # grid
    for xline in major_ticks:
        ax.vlines(xline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='gray', linestyle='--', linewidth=1,
                  zorder=1, alpha=0.9)
    for yline in major_ticks:
        ax.hlines(yline, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='gray', linestyle='--', linewidth=1,
                  zorder=1, alpha=0.9)

    ax.set_aspect(aspect='equal')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.tick_params('x', length=50)
    ax.tick_params('y', length=50)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)

    displacement_factor = 2
    for minor_tick in ax.xaxis.get_minorticklocs():
        ax.text(minor_tick, ax.get_ylim()[0] - displacement_factor, minor_tickslabel_x[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='center', rotation=90)
    for minor_tick in ax.yaxis.get_minorticklocs():
        ax.text(ax.get_xlim()[0] - displacement_factor, minor_tick, minor_tickslabel_y[minor_ticks.index(minor_tick)],
                fontsize=8, color='gray', ha='center', va='center')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.draw()
    plt.savefig(f'Images/Supermatrix_Pixel_Plot_l={l}_n={n}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_supermatrix_l5_n12(l, n, linthresh, trunc=None):
    # load supermatrix
    name_string = f'supermatrix_array_{l}_{n}.txt'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices', name_string)
    supermatrix_array = np.loadtxt(DATA_DIR, delimiter=' ')
    supermatrix_array = supermatrix_array.astype(float)

    # Load index_map
    name_string = f'index_map_supermatrix_array_{l}_{n}.txt'
    data_dir = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Supermatrices',
                            name_string)
    index_map = load_index_map_from_file(data_dir)

    if trunc is not None:
        index_map_l = np.unique(index_map['l'])[:-trunc]
        mask = np.isin(index_map['l'], index_map_l) & np.isin(index_map['lprime'], index_map_l)
        true_indices = np.argwhere(mask[0] == False)
        index_map = np.delete(index_map, true_indices, axis=0)
        index_map = np.delete(index_map, true_indices, axis=1)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=0)
        supermatrix_array = np.delete(supermatrix_array, true_indices, axis=1)

    # print elements above threshold:
    thresh = linthresh
    data_above = []
    for i in range(len(supermatrix_array)):
        for j in range(len(supermatrix_array[i])):
            if np.abs(supermatrix_array[i][j]) > thresh:
                data_above.append((supermatrix_array[i][j], index_map[i][j]))
    sorted_data_above = sorted(data_above, key=lambda x: x[1][0])
    for value, index in sorted_data_above:
        print(value, index)

    # linthresh = np.min(np.abs(supermatrix_array[supermatrix_array != 0]))
    # print(linthresh)
    vmax = np.max(np.abs(supermatrix_array))
    # norm = SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, linscale=2)
    fig, ax = plt.subplots(figsize=(13, 13))
    # Slice seismic colorbar
    seismic = plt.get_cmap("seismic")
    red_half = seismic(np.linspace(0.5, 1, 1024))
    red_cmap = ListedColormap(red_half)

    cax = ax.pcolormesh(supermatrix_array, cmap=red_cmap, alpha=1)
    # Add colorbar
    cb = plt.colorbar(cax, ax=ax, label=f'Supermatrixelement in µHz$^2$', pad=0.02, shrink=0.84)
    cb.set_label(label=f'Supermatrixelement in µHz$^2$', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    plt.title(f'Supermatrix for quasi-degenerate multiplet around l={l}, n={n}', fontsize=16)

    major_ticks = []
    minor_ticks = []
    tickslabel_x = []
    tickslabel_y = []
    minor_tickslabel_x = []
    minor_tickslabel_y = []
    for i in range(len(index_map[0])):
        column = index_map[0][i]
        l_co = column['l']
        minor_ticks.append(i + 0.5)
        minor_tickslabel_x.append(f'm={column["m"]}')
        minor_tickslabel_y.append(f'm\'={column["m"]}')

        if l_co == -column['m']:
            major_ticks.append(i)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')
        elif i == len(index_map[0]) - 1:
            major_ticks.append(i + 1)
            tickslabel_x.append(f'l={l_co}, n={column["n"]}')
            tickslabel_y.append(f'l\'={l_co}, n\'={column["n"]}')

    for i in range(len(major_ticks) - 1):
        x_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        ax.text(x_mid, ax.get_ylim()[0] - 0.075 * (ax.get_ylim()[1] - ax.get_ylim()[0]), tickslabel_x[i],
                ha='center', va='center', fontsize=16)  # adjust factor for y position

    for i in range(len(major_ticks) - 1):
        y_mid = (major_ticks[i] + major_ticks[i + 1]) / 2
        ax.text(ax.get_xlim()[0] - 0.075 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_mid, tickslabel_y[i],
                ha='center', va='center', fontsize=16, rotation=90)  # adjust factor for y position

    # grid
    for xline in major_ticks:
        ax.vlines(xline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='gray', linestyle='--', linewidth=1,
                  zorder=1, alpha=0.9)
    for yline in major_ticks:
        ax.hlines(yline, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='gray', linestyle='--', linewidth=1,
                  zorder=1, alpha=0.9)

    ax.set_aspect(aspect='equal')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.tick_params('x', length=50)
    ax.tick_params('y', length=50)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)

    displacement_factor = 0.35
    for minor_tick in ax.xaxis.get_minorticklocs():
        ax.text(minor_tick, ax.get_ylim()[0] - displacement_factor, minor_tickslabel_x[minor_ticks.index(minor_tick)],
                fontsize=10, color='gray', ha='center', va='center', rotation=90)
    for minor_tick in ax.yaxis.get_minorticklocs():
        ax.text(ax.get_xlim()[0] - displacement_factor, minor_tick, minor_tickslabel_y[minor_ticks.index(minor_tick)],
                fontsize=10, color='gray', ha='center', va='center')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.draw()
    plt.savefig(f'Images/Supermatrix_Pixel_Plot_l={l}_n={n}_linear.png', dpi=300, bbox_inches='tight')
    plt.show()



def main():
    # Initialize configuration handler
    config = ConfigHandler("config.ini")

    # Initialize path to Frequency_shifts folder
    freq_dir = os.path.join(os.path.dirname(__file__), 'Output', config.get("ModelConfig", "model_name"), 'FrequencyShifts')

    ######################################
    # Plot frequency shifts
    plot_freq_shifts = False

    # Define Parameters; change this as needed
    rel_shifts = True   # Use this for rel. freq shift over m/l
    abs_shifts = False   # Only useful for low l
    combined = False     # Only useful for low l
    l, n = 5, 6
    save = False

    # Load eigentag
    eigentag = config.get("Eigenspace", "eigenspace_tag")
    if eigentag is None or eigentag == 'Full':
        name_string = f'freq_shifts_{l}_{n}_full.txt'
    elif eigentag == 'FirstApprox':
        name_string = f'freq_shifts_{l}_{n}_first_approx.txt'
    elif eigentag == 'SelfCoupling':
        name_string = f'freq_shifts_{l}_{n}_self_coupling.txt'
    else:
        # Raise an error if the eigen_tag is invalid
        raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')
    # As default use name with eigentag from config, but the file_name can be changed as needed
    file_name = name_string

    if plot_freq_shifts:
        # Path can be adjusted as needed; default is Output/model_name/FrequencyShifts
        path = os.path.join(freq_dir, file_name)
        plot_frequency_shifts(l, n, plot_relative_shifts=rel_shifts, plot_absolute_shifts=abs_shifts, plot_combined=combined, path=path, save=save)

    ######################################
    # Plot overlaid frequency shifts
    plot_overlaid = True
    l, n = [], []
    for i in range(2,60):
        l.append(i)
        n.append(10)
    #l = [5, 6, 7, 8, 9, 10]
    #n = [5, 5, 5, 5, 5, 5]
    save = False
    if plot_overlaid:
        plot_frequency_shifts_overlaid(l, n, freq_dir, eigentag, save)

    ######################################
    # Plot output summary: Plots frequency shift over multiplet frequency or lower turning point (if given)
    output_summary = False
    comp_rt = False
    r_thresh = 0.863
    save = True

    # Compute lower turning points
    # Initialize stellar model (MESA data)
    mesa_data = radial_kernels.MesaData(config=config)
    summary_dir = os.path.join(os.path.dirname(__file__), 'Data', 'GYRE', config.get("StellarModel", "summary_GYRE_path"))
    if comp_rt:
        r_t = lower_turning_points(summary_dir, mesa_data)
    else:
        r_t = None

    if output_summary:
        # Create tuple list of l and n
        l_list, n_list = [], []
        for l in range(2, 140+1):
            for n in range(0, 35+1):
                criterium_large = (128 <= l < 138 and n >= 13) or (
                        118 <= l < 128 and n >= 12) or (
                                          108 <= l < 118 and n >= 11) or (
                                          98 <= l < 108 and n >= 10) \
                                  or (88 <= l < 98 and n >= 9) or (
                                          79 <= l < 88 and n >= 8) or (
                                          69 <= l < 79 and n >= 7) \
                                  or (60 <= l < 69 and n >= 6) or (
                                          51 <= l < 60 and n >= 5) or (
                                          42 <= l < 51 and n >= 4) \
                                  or (31 <= l < 42 and n >= 3) or (
                                          20 <= l < 31 and n >= 2) or l <= 19 and n >= 1
                freq_temp = GMEP.frequencies_GYRE(l,n)
                if eigentag is None or eigentag == 'Full':
                    name_string = f'freq_shifts_{l}_{n}_full.txt'
                elif eigentag == 'FirstApprox':
                    name_string = f'freq_shifts_{l}_{n}_first_approx.txt'
                elif eigentag == 'SelfCoupling':
                    name_string = f'freq_shifts_{l}_{n}_self_coupling.txt'
                else:
                    # Raise an error if the eigen_tag is invalid
                    raise ValueError('Unknown eigenspace tag. Use "Full", "FirstApprox" or "SelfCoupling".')
                file_path = os.path.join(freq_dir, name_string)
                if freq_temp is not None and criterium_large and os.path.exists(file_path):
                    l_list.append(l)
                    n_list.append(n)
        plot_output_summary(l_list, n_list, freq_dir, eigentag, r_t, r_thresh, save)


if __name__ == "__main__":
    main()
