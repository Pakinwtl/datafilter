import os, shutil
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from dataclasses import dataclass
from statsmodels.nonparametric.smoothers_lowess import lowess

# FUNCTION: folder cleaning
def folder_clean():
    print("Clear Savefig Folder")
    folder = glob.glob('./code/Savefig/*')
    for filename in folder:
        os.remove(filename)


# FUNCTION: import data into DATAFRAME #
def data_import(args):
    df = pd.read_excel(f'{args.data_path}{args.trial}/{args.file}')
    wavelength = df.columns[0]
    filtered_df = df[(df[wavelength] > 450) & (df[wavelength] < 950)]
    wavelength = filtered_df.iloc[:, 0]
    raw_intensity = filtered_df.iloc[:, 1:]

    return filtered_df, wavelength, raw_intensity

def data_import2(files_path):
    df = pd.read_excel(files_path)
    wavelength = df.columns[0]
    filtered_df = df[(df[wavelength] > 450) & (df[wavelength] < 950)]
    wavelength = filtered_df.iloc[:, 0]
    raw_intensity = filtered_df.iloc[:, 1:]

    return filtered_df, wavelength, raw_intensity
#################################################


# FUNCTION: DATA PROCESSING #
def data_processing(args, df, wavelength, intensity):

    # INTENSITY #
    raw_i0 = df.iloc[:, 1]  #AIR REF
    raw_i1 = df.iloc[: ,2:]  #DATA FROM 20-100%
    smoothed_intensity = pd.DataFrame(index=wavelength)
    for col in intensity.columns:
        smoothed = lowess(intensity[col], wavelength, frac=0) # แก้ตรงนี้เป็น 0 ไปเหลือ smooth จุด trans ----- 9/15/2024
        smoothed_intensity[col] = smoothed[:, 1]

    sm_i0 = smoothed_intensity.iloc[:, 0] # SMOOTH AIR REF
    sm_i1 = smoothed_intensity.iloc[:, 1:] # SMOOTH INTENSITY

    # TRANSMITTANCE #
    raw_trans = pd.DataFrame(index=wavelength)
    sm_trans = pd.DataFrame(index=wavelength)

    for col in raw_i1.columns:
        raw_trans[col] = raw_i1[col]/raw_i0
    for col in sm_i1.columns:
        sm_trans[col] = sm_i1[col]/sm_i0

    for col in sm_trans.columns:
        smoothed = lowess(sm_trans[col], wavelength, frac=args.smooth_factor) #----------
        sm_trans[col] = smoothed[:, 1]
    return sm_trans, raw_trans, raw_i1, sm_i1
#################################################

# PLOT INTENSITY OVER WAVELENGTH
def graph_plot(wavelength, raw_i, sm_i, sm_T, sname1, sname2):
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    for col in raw_i.columns:
        ax[0].plot(wavelength, raw_i[col])
    ax[0].legend(['500ppm', '400ppm', '300ppm', '200ppm', '100ppm'])
    ax[0].title.set_text('raw intensity')
    for col in sm_i.columns:
        ax[1].plot(wavelength, sm_i[col])
    ax[1].legend(['500ppm', '400ppm', '300ppm', '200ppm', '100ppm'])
    ax[1].title.set_text('smooth intensity')
    plt.savefig(spath + sname1)
    plt.close()

    for col in sm_T.columns:
        plt.plot(wavelength, sm_T[col])
    plt.legend(['500ppm', '400ppm', '300ppm', '200ppm', '100ppm'])
    plt.grid(True)
    plt.savefig(spath + sname2)
    plt.close()

# FUNCTION: LOCATE PICTURE FOR FINAL RESULT #
def image_plot(x_value, y_value, sname, data, wavelength, args):
    for i in range(y_value.shape[0]):
        plt.xlim(np.min(x_value[i])-4, np.max(x_value[i])+4)
        for col in data.columns:
            plt.plot(wavelength, data[col])
        plt.title(f'Trend of {args.file} {i+1}')
        plt.xlabel('wavelength')
        plt.legend(['500ppm', '400ppm', '300ppm', '200ppm', '100ppm'])
        plt.grid(True)
        plt.savefig(spath + f'{sname}_{i+1}.png')
        plt.close()

# FUNCTION: Ploting WAVELENGTH OVER CONCENTRATION #
def ploting(x_value, y_value, args, sname):
    for i in range(y_value.shape[0]):
        plt.scatter(args.concentrations, x_value[i])
        z = np.polyfit(args.concentrations, x_value[i], 1)
        p = np.poly1d(z)
        plt.annotate(f'r value: {r_value[i]:.3f}', 
                     xy=(args.concentrations[3], p(args.concentrations[3])), 
                     xytext=(10, 10), 
                     textcoords='offset points', 
                     fontsize=8,
                     color='blue')
        plt.plot(args.concentrations, p(args.concentrations))
        plt.title(f'Trend of {args.file}  {i+1} ')
        plt.grid(True)
        plt.savefig(spath + f'{sname}_{i+1}.png')
        plt.close()
        
def save_value(y_value, sname, args):
    try:
        print('save the value')
        df = pd.DataFrame(y_value, columns=args.concentrations)
        df.to_csv(spath + sname)
    except ValueError:
        print("There is no Value that exceed 0.8")
#################################################


# FUNCTION: data slicing #
def data_windowing(series, args):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(args.window_size, shift=args.stride, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(args.window_size + 1))
    return dataset
#################################################


# FUNCTION: find min values #
def find_min(data, df):
    print("Start filter min value and trend")
    min_value_list, min_wv_list = [], []
    data_array = np.array(data)
    for i in range(len(data_array)):
        min_values = np.min(data_array[i], axis=0)
        min_index = np.argmin(data_array[i], axis=0)
        min_wv = sm_trans.index[min_index + i].to_numpy()

        if np.any(min_index == 0) or np.any(min_index == args.window_size - 1) or not np.all(np.diff(min_values) > 0):
            continue
    
        min_value_list.append(min_values)
        min_wv_list.append(min_wv)

    min_values_array = np.array(min_value_list)
    min_wv_array = np.array(min_wv_list)

    ans_value, ans_index = np.unique(min_values_array, axis=0, return_index=True)
    ans_wv = min_wv_array[ans_index]
    print(ans_value)

    return ans_value[:, ::-1], ans_wv[:, ::-1], df.columns[:1:-1]
#################################################


# FUNCTION: filter r value #
def find_r(args, y_value, x_value):
    print(f"filter R value exceeding {args.r_value}")
    filtered_x, filtered_y, r_value, all_r = [], [], [], []
    for i in range(y_value.shape[0]):
        _, _, r, _, _ = linregress(args.concentrations, x_value[i])
        all_r.append(abs(r))
        if abs(r) >= args.r_value:
            r_value.append(abs(r))
            filtered_x.append(x_value[i])
            filtered_y.append(y_value[i])
    if len(r_value) == 0:
        print(f"--------------There is no R>{args.r_value}--------------------!!!!!!!!!!!!!!!!!!!!")
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)

    return filtered_x, filtered_y, r_value, all_r
#################################################




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=str, default='analyze_trial1', help='Number of the trial')
    parser.add_argument('--file', type=str, default='5min.xlsx', help='Name of the trial file')
    parser.add_argument('--data-path', type=str, default=r'./SeniorProj/Acetone_AI_data_15mins_runtime/analysis_test/', help='Path to Data')
    parser.add_argument('--window-size', type=int, default=50, help="Size of Window")
    parser.add_argument('--batch-size', type=int, default=1, help='Number of Batch')
    parser.add_argument('--stride', type=int, default=1, help='Number of Stride')
    parser.add_argument('--smooth-factor', type=float, default=0.00522, help='Value of Smooth factor')
    parser.add_argument('--concentrations', type=int, default=np.array([20, 40, 60, 80, 100]), help='Concentration Percentage')
    parser.add_argument('--r-value', type=float, default=0.7  , help='Minimum R value')
    parser.add_argument('--start1', action='store_true', help='Start function')
    parser.add_argument('--start2', action='store_true', help='Start function')
    args = parser.parse_args()

    global spath  
    spath = f'./code/Savefig/'
    os.makedirs(spath, exist_ok=True)

    if args.start1:
            
        print("Start Process")

        folder_clean()
        # Data import #
        df, wavelength, intensity = data_import(args)

        print(f'Trial number {args.trial}')
        print(f'file number {args.file}')
        print(f'window size : {str(args.window_size)}')
        print(f'smooth factor : {str(args.smooth_factor)}')

        # smoothing process #
        sm_trans, raw_trans, raw_i, sm_i = data_processing(args, df, wavelength, intensity)

        graph_plot(wavelength, raw_i, sm_i, sm_trans, 'intensity.png', 'transmittance.png')

        # iterating process #
        big_window = data_windowing(sm_trans, args)
        dataset = [window.numpy() for window in big_window]

        # FIND MIN #
        min_value, min_wavelength, columns = find_min(dataset, df)
        
        # FIND R VALUE #
        filtered_wv, filtered_min, r_value, all_r = find_r(args, min_value, min_wavelength)

    
        print('VALUE BEFORE FILTER')
        print(f'WAVELENGTH : {min_wavelength}')
        print(f'all R value :{all_r}')

        print('concentrations : 100 200 300 400 500')
        print(f'MIN VALUE : {filtered_min}')
        print(f'WAVELENGTH : {filtered_wv}')

        ploting(filtered_wv, filtered_min, args, 'r_value')
        image_plot(filtered_wv, filtered_min, 'image', sm_trans, wavelength, args)
        save_value(filtered_wv, 'save_wavelength.csv', args)

        print('Process END')
    
