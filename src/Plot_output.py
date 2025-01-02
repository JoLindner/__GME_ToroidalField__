import numpy as np
import os
import matplotlib.pyplot as plt
import GeneralMatrixElements_Parallel as GMEP
from matplotlib.ticker import ScalarFormatter


def plot_frequencs_shifts(l,n,plot_relative_shifts=True, plot_absolute_shifts=True, plot_combined=False):
    try:
        name_string = f'freq_shifts_{l}_{n}_first_approx.txt'
        #Change this path to the path where the results are stored
        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_combined', 'Frequency_shifts', name_string)
    except (FileNotFoundError, KeyError, StopIteration) as e:
        return f'File {DATA_DIR} does not exist'

    data = np.loadtxt(DATA_DIR, comments='#', delimiter=' ')
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
        #plt.legend()
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
        plt.savefig(f'Images/FrequencyShifts_l{l}_n{n}.pdf', dpi=300, bbox_inches='tight')
        plt.show()


def plot_frequency_shifts_overlaid(l_list,n_list):
    try:
        #Change this path to the path where the results are stored
        DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_1708_first_run_part3', 'Frequency_shifts')
    except (FileNotFoundError, KeyError, StopIteration) as e:
        return f'File {DATA_DIR} does not exist'

    name_string_list=[]
    data_list=[]
    freq_shift_list=[]
    m_list=[]
    freq_list=[]
    relative_shift_list=[]
    marker_list=['x','o','+']
    plt.figure(figsize=(11, 8))
    for value in range(len(l_list)):
        name_string_list.append(f'freq_shifts_{l_list[value]}_{n_list[value]}.txt')
        data_list.append(np.loadtxt(os.path.join(DATA_DIR, name_string_list[value]), comments='#', delimiter=' '))
        freq_shift_list.append(data_list[value][:,0]) #nHz
        m_list.append(data_list[value][:,3])
        freq_list.append(GMEP.frequencies_GYRE(l_list[value],n_list[value])*10**3)   #nHz
        relative_shift_list.append(freq_shift_list[value]/freq_list[value])
        plt.scatter(m_list[value]/l_list[value],relative_shift_list[value], marker=marker_list[value], label=f'l={l_list[value]},n={n_list[value]}')
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
    plt.title(r'Relative frequency shifts of multiplet $l=5,n={5,12,18}$', fontsize=16)
    plt.legend()
    plt.show()

def plot_approximations(l_list,n_list):
    #load data
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Frequency_shifts')
    frequencies, frequencies_first_approx, frequencies_second_approx =[], [], []
    for i in range(len(l_list)):
        name_string=f'freq_shifts_{l_list[i]}_{n_list[i]}.txt'
        frequencies.append(np.loadtxt(os.path.join(DATA_DIR, name_string), comments='#', delimiter=' '))
        name_string=f'freq_shifts_{l_list[i]}_{n_list[i]}_first_approx.txt'
        frequencies_first_approx.append(np.loadtxt(os.path.join(DATA_DIR, name_string), comments='#', delimiter=' '))
        name_string=f'freq_shifts_{l_list[i]}_{n_list[i]}_second_approx.txt'
        frequencies_second_approx.append(np.loadtxt(os.path.join(DATA_DIR, name_string), comments='#', delimiter=' '))

    #calculate differences
    diff={'diff_first_approx': [], 'diff_second_approx': []}
    for freq, freq_a1, freq_a2 in zip(frequencies, frequencies_first_approx, frequencies_second_approx):
        for f,f1,f2 in zip(freq, freq_a1, freq_a2):
            diff['diff_first_approx'].append({'diff': f[0] - f1[0], 'l': f[1], 'n': f[2], 'm': f[3]})
            diff['diff_second_approx'].append({'diff': f[0] - f2[0], 'l': f[1], 'n': f[2], 'm': f[3]})

    #plot data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
    unique_n = sorted(set(d['n'] for d in diff['diff_first_approx']))
    color=['b','r','g']
    i=0
    rms = {'rms_first_approx': [], 'rms_second_approx': []}
    for n in unique_n:
        first_approx = [d for d in diff['diff_first_approx'] if d['n'] == n]
        second_approx = [d for d in diff['diff_second_approx'] if d['n'] == n]
        m_l_first = [d['m'] / d['l'] for d in first_approx]
        diff_first = [d['diff'] for d in first_approx]

        m_l_second = [d['m'] / d['l'] for d in second_approx]
        diff_second = [d['diff'] for d in second_approx]

        #compute RMS
        rms['rms_first_approx'].append({'rms': np.sqrt(np.mean(np.array(diff_first) ** 2)), 'n': n})
        rms['rms_second_approx'].append({'rms': np.sqrt(np.mean(np.array(diff_second) ** 2)), 'n': n})

        if n==6:
            ax2.scatter(m_l_first, diff_first, marker='s', color=color[i], label=f'l=5, n={int(n)} (Approx1)', alpha=1, s=100, facecolor='none', edgecolors=color[i], linewidths=2,)
            ax1.scatter(m_l_first, diff_second, marker='o', color=color[i], label=f'l=5, n={int(n)} (Approx2)', alpha=0.75, s=100)
            i+=1
            continue
        # Plot for first approximation
        ax2.scatter(m_l_first, diff_first, marker='s', color=color[i], label=f'l=5, n={int(n)} (Approx1)', alpha=1, s=100, facecolor='none', edgecolors=color[i], linewidths=2)
        # Plot for second approximation
        ax2.scatter(m_l_second, diff_second, marker='o', color=color[i], label=f'l=5, n={int(n)} (Approx2)', alpha=0.75, s=100)
        i+=1
    ax2.set_xlabel(f'$m/l$', fontsize=14)
    ax2.set_ylabel(r'$\delta\nu_{full}-\delta\nu_{approx}$ in nHz', fontsize=14)
    ax1.set_ylabel(r'$\delta\nu_{full}-\delta\nu_{approx}$ in nHz', fontsize=14)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_title(r'Differences in absolute frequency shifts of multiplets $l=5, n=\{6,12,18\}$', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    handles, labels = ax2.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels),key=lambda x: (x[1].endswith('(Approx2)'), x[1].endswith('(Approx1)')))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    ax2.legend(sorted_handles, sorted_labels, loc='upper left', fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'Images/Freq_shift_diffs.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(rms['rms_first_approx'])
    print(rms['rms_second_approx'])


def plot_approximations_defense(l_list,n_list):
    #load data
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output', 'Frequency_shifts')
    frequencies, frequencies_first_approx, frequencies_second_approx =[], [], []
    for i in range(len(l_list)):
        name_string=f'freq_shifts_{l_list[i]}_{n_list[i]}.txt'
        frequencies.append(np.loadtxt(os.path.join(DATA_DIR, name_string), comments='#', delimiter=' '))
        name_string=f'freq_shifts_{l_list[i]}_{n_list[i]}_first_approx.txt'
        frequencies_first_approx.append(np.loadtxt(os.path.join(DATA_DIR, name_string), comments='#', delimiter=' '))
        name_string=f'freq_shifts_{l_list[i]}_{n_list[i]}_second_approx.txt'
        frequencies_second_approx.append(np.loadtxt(os.path.join(DATA_DIR, name_string), comments='#', delimiter=' '))

    #calculate differences
    diff={'diff_first_approx': [], 'diff_second_approx': []}
    for freq, freq_a1, freq_a2 in zip(frequencies, frequencies_first_approx, frequencies_second_approx):
        for f,f1,f2 in zip(freq, freq_a1, freq_a2):
            diff['diff_first_approx'].append({'diff': f[0] - f1[0], 'l': f[1], 'n': f[2], 'm': f[3]})
            diff['diff_second_approx'].append({'diff': f[0] - f2[0], 'l': f[1], 'n': f[2], 'm': f[3]})

    #plot data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharex=False)
    unique_n = sorted(set(d['n'] for d in diff['diff_first_approx']))
    color=['b','r','g']
    i=0
    rms = {'rms_first_approx': [], 'rms_second_approx': []}
    for n in unique_n:
        first_approx = [d for d in diff['diff_first_approx'] if d['n'] == n]
        second_approx = [d for d in diff['diff_second_approx'] if d['n'] == n]
        m_l_first = [d['m'] / d['l'] for d in first_approx]
        diff_first = [d['diff'] for d in first_approx]

        m_l_second = [d['m'] / d['l'] for d in second_approx]
        diff_second = [d['diff'] for d in second_approx]

        #compute RMS
        rms['rms_first_approx'].append({'rms': np.sqrt(np.mean(np.array(diff_first) ** 2)), 'n': n})
        rms['rms_second_approx'].append({'rms': np.sqrt(np.mean(np.array(diff_second) ** 2)), 'n': n})

        if n==6:
            ax2.scatter(m_l_first, diff_first, marker='s', color=color[i], label=f'l=5, n={int(n)} (Approx1)', alpha=1, s=100, facecolor='none', edgecolors=color[i], linewidths=2,)
            ax1.scatter(m_l_first, diff_second, marker='o', color=color[i], label=f'l=5, n={int(n)} (Approx2)', alpha=0.75, s=100)
            i+=1
            continue
        # Plot for first approximation
        ax2.scatter(m_l_first, diff_first, marker='s', color=color[i], label=f'l=5, n={int(n)} (Approx1)', alpha=1, s=100, facecolor='none', edgecolors=color[i], linewidths=2)
        # Plot for second approximation
        ax2.scatter(m_l_second, diff_second, marker='o', color=color[i], label=f'l=5, n={int(n)} (Approx2)', alpha=0.75, s=100)
        i+=1
    ax1.set_xlabel(f'$m/l$', fontsize=14)
    ax2.set_xlabel(f'$m/l$', fontsize=14)
    ax2.set_ylabel(r'$\delta\nu_{full}-\delta\nu_{approx}$ in nHz', fontsize=14)
    ax1.set_ylabel(r'$\delta\nu_{full}-\delta\nu_{approx}$ in nHz', fontsize=14)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    fig.suptitle(r'Differences in absolute frequency shifts of multiplets $l=5, n=\{6,12,18\}$', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    handles, labels = ax2.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels),key=lambda x: (x[1].endswith('(Approx2)'), x[1].endswith('(Approx1)')))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    ax2.legend(sorted_handles, sorted_labels, loc='lower left', fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'Images/Freq_shift_diffs_defense.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(rms['rms_first_approx'])
    print(rms['rms_second_approx'])


def plot_output_summary(l_list, n_list):
    #load data
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_combined', 'Frequency_shifts')
    frequency_shifts = []
    averaged_shifts = []
    for i in range(len(l_list)):
         name_string=f'freq_shifts_{l_list[i]}_{n_list[i]}_first_approx.txt'
         file_path = os.path.join(DATA_DIR, name_string)
         try:
            #print(i,l_list[i], n_list[i])
            frequency_shifts.append(np.loadtxt(file_path, comments='#', delimiter=' '))
            averaged_shifts_dict = {'l': l_list[i], 'n': n_list[i], 'freq': GMEP.frequencies_GYRE(l_list[i], n_list[i]), 'av_shift': np.mean(frequency_shifts[i][:, 0])}
            averaged_shifts.append(averaged_shifts_dict)
         except FileNotFoundError:
             #print(f"Error: Unable to load {file_path}")
             continue

    freq_values = [entry['freq'] for entry in averaged_shifts]
    av_shifts = [entry['av_shift'] for entry in averaged_shifts]
    l_values = [entry['l'] for entry in averaged_shifts]

    #plot data
    plt.figure(figsize=(12, 8))
    plt.xlabel('$\\nu$ in µHz', size=16)
    plt.ylabel(r'$\delta\nu$ in nHz', size=16)
    scatter = plt.scatter(freq_values, av_shifts, c=l_values, s=5, cmap='gnuplot', alpha=0.75)
    colorbar = plt.colorbar(scatter, label='$l$')
    colorbar.set_label(label='harmonic degree $l$', size=16)
    colorbar.ax.tick_params(labelsize=14)

    unique_l = np.unique(l_values)
    colors = scatter.cmap(scatter.norm(unique_l))
    for l_value, color in zip(unique_l, colors):
        subset_indices = [i for i, val in enumerate(l_values) if val == l_value]
        plt.plot(np.array(freq_values)[subset_indices], np.array(av_shifts)[subset_indices], color=color, alpha=0.5, linewidth=1)

    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('Images/Freq_shift_summary.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    #print(next((entry['av_shift'] for entry in averaged_shifts if entry['l'] == 5 and entry['n'] == 8)))

def main():
    l=[5,5,5]
    n=[6,12,18]
    #plot_frequency_shifts_overlaid(l,n)
    #plot_approximations_defense(l,n)

    #l=70
    #n=8
    #plot_frequencs_shifts(l,n, True, False, False)

    #Plot output summary
    #Create tuple list of l and n
    l_list, n_list = [], []
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'Output_combined', 'Frequency_shifts')

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
            freq_temp = GMEP.frequencies_GYRE(l,n)
            name_string = f'freq_shifts_{l}_{n}_first_approx.txt'
            file_path = os.path.join(DATA_DIR, name_string)

            if freq_temp is not None and criterium_large and os.path.exists(file_path):
                l_list.append(l)
                n_list.append(n)

    plot_output_summary(l_list, n_list)


    '''
    l=5
    n=6
    plot_frequencs_shifts(l,n, False, False, True)
    l = 5
    n = 18
    plot_frequencs_shifts(l,n, False, False, True)

    l = 5
    n = 12
    plot_frequencs_shifts(l,n, False, False, True)
    '''

if __name__ == "__main__":
    main()