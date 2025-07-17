import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import h5py
from concurrent.futures import ProcessPoolExecutor
import glob

# Function to compute MSD
def MSD(x):
    Nt = len(x)
    dsq = np.square(x)
    SumSq = 2 * np.dot(x, x)
    Sab = np.correlate(x, x, mode='full')[Nt-1:]
    msd = np.zeros(Nt - 1)
    for m in range(1, Nt - 1):
        SumSq -= dsq[m - 1] + dsq[Nt - m]
        msd[m] = (SumSq - 2 * Sab[m]) / (Nt - m)
    return msd

# Function to process each file
def processFile(file_path):
    with h5py.File(file_path, 'r') as f:
        timeMD = np.array(f['time'])
        XMD = np.array(f['trajectoryX'])
        YMD = np.array(f['trajectoryY'])
        
    NframesMD = XMD.shape[0]
    Natoms = XMD.shape[1]

    # Sample trajectories
    Ngap = 1
    time = timeMD[::Ngap]
    X = XMD[::Ngap]
    Y = YMD[::Ngap]
    
    Nframes = X.shape[0]

    # Compute MSDs for all atoms with parallelization
    with ProcessPoolExecutor() as executor:
        MSDx = list(executor.map(MSD, [X[:, i] for i in range(Natoms)]))
        MSDy = list(executor.map(MSD, [Y[:, i] for i in range(Natoms)]))
    
    # Compute averaged MSDs in the x-y plane
    Nt = Nframes // 10
    MSDxAv = np.mean(MSDx, axis=0)
    MSDyAv = np.mean(MSDy, axis=0)
    MSDxyAv = (MSDxAv + MSDyAv) / 2

    # Prepare MSDs for plots
    MSDxyAvTable = np.column_stack((time[:Nt], MSDxyAv[:Nt]))

    # Exclude data points outside 10 ps to 80 ps for fitting
    mask = (MSDxyAvTable[:, 0] >= 10) & (MSDxyAvTable[:, 0] <= 80)
    MSDxyAvTable_fit = MSDxyAvTable[mask]

    # Fit MSD model only within 10 ps - 80 ps
    epsilon = 1e-7
    def MSDmodel(t, D_alpha, alpha):
        return 2 * D_alpha * (t + epsilon)**alpha
    
    popt, _ = curve_fit(MSDmodel, MSDxyAvTable_fit[:, 0], MSDxyAvTable_fit[:, 1])

    return MSDxyAvTable, MSDxyAvTable_fit, popt

if __name__ == "__main__":
    # Set directory and find unique prefixes (e.g., 316-*, 290-*, etc.)
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    
    # Find all unique prefixes based on existing files
    files = glob.glob("*-lipid-coord-*.h5")
    unique_prefixes = sorted(set(f.split('-')[0] for f in files))  # Extract unique prefixes

    for prefix in unique_prefixes:
        print(f"\nProcessing dataset: {prefix}-*\n")

        # Get all files matching this prefix
        TrajectoryFiles = sorted(glob.glob(f"{prefix}-lipid-coord-*.h5"))

        # Lists to store results
        fitParamsList = []
        all_MSDs = []
        time_values = None

        for i, file in enumerate(TrajectoryFiles):
            MSDxyAvTable, MSDxyAvTable_fit, popt = processFile(file)
            fitParamsList.append([i + 1, round(popt[0], 9), round(popt[1], 9)])  # Store set number instead of filename

            # Store MSD values
            if time_values is None:
                time_values = MSDxyAvTable[:, 0]  # Use time from first dataset
            all_MSDs.append(MSDxyAvTable[:, 1])  # Store MSD values

        # Compute average MSD and standard deviation
        avg_MSD = np.mean(all_MSDs, axis=0)
        std_MSD = np.std(all_MSDs, axis=0)

        # Fit the averaged MSD
        mask_avg = (time_values >= 10) & (time_values <= 80)
        time_fit_avg = time_values[mask_avg]
        MSD_fit_avg = avg_MSD[mask_avg]

        epsilon = 1e-7
        def MSDmodel(t, D_alpha, alpha):
            return 2 * D_alpha * (t + epsilon)**alpha

        popt_avg, _ = curve_fit(MSDmodel, time_fit_avg, MSD_fit_avg)

        # Save the average MSD data
        np.savetxt(f"{prefix}-average_MSD.txt", np.column_stack((time_values, avg_MSD, std_MSD)), 
                   header="Time(ps)  Average_MSD  Std_Deviation", fmt="%.6f")

        # Save the fitting data shown in the figure
        np.savetxt(f"{prefix}-fit_curve.txt", np.column_stack((time_fit_avg, MSDmodel(time_fit_avg, *popt_avg))), 
                   header="Time(ps)  Fitted_MSD", fmt="%.6f")

        # Create plot for only average MSD
        plt.figure(figsize=(10, 8))
        
        # Plot the average MSD
        plt.plot(time_values, avg_MSD, linestyle='-', marker='o', color='black', markersize=6, label='Average MSD', linewidth=2)

        # Plot the fit for the average MSD
        plt.plot(time_fit_avg, MSDmodel(time_fit_avg, *popt_avg), color='black', linewidth=3, linestyle="--", label="Fit (Average)")

        # Set log scale
        plt.xscale('log')
        plt.yscale('log')

        # Make axis lines thicker
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(3)

        # Set tick labels bold and larger
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')

        # Make major and minor ticks thicker and longer
        ax.xaxis.set_tick_params(which='both', width=2, length=8)  
        ax.yaxis.set_tick_params(which='both', width=2, length=8)

        # Set axis labels in bold
        plt.xlabel(r'$\mathbf{t\ [ps]}$', fontsize=16, fontweight='bold')
        plt.ylabel(r'$\mathbf{MSD\ [Å²]}$', fontsize=16, fontweight='bold')

        # Save the plot
        plt.legend(fontsize=12)
        plt.savefig(f"{prefix}-msd_average_fit.png", dpi=300)
        plt.show()

        # Save the reformatted fit parameters
        fitParamsFile = f"{prefix}-fit_parameters_10_80ps.txt"
        df_fit = pd.DataFrame(fitParamsList, columns=['Set', 'D_alpha', 'Alpha'])

        # Compute Mean and Std Dev **only** for D_alpha and Alpha
        df_fit.loc[len(df_fit)] = ["Mean", round(df_fit["D_alpha"].mean(), 9), round(df_fit["Alpha"].mean(), 9)]
        df_fit.loc[len(df_fit)] = ["Std Dev", round(df_fit["D_alpha"].std(), 9), round(df_fit["Alpha"].std(), 9)]

        # Save to file
        df_fit.to_csv(fitParamsFile, sep='\t', index=False, float_format='%.9f')

        print(f"Saved: {prefix}-average_MSD.txt, {prefix}-fit_curve.txt, {prefix}-fit_parameters_10_80ps.txt, {prefix}-msd_average_fit.png\n")
