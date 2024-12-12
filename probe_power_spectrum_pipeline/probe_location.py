import matplotlib.pyplot as plt
from pathlib import Path


import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.widgets as sw

import numpy as np
import  multitaper
from tqdm import tqdm
import pandas as pd
import os
from scipy import signal

# INPUT = Path('/ceph/sjones/projects/sequences/NPX_DATA/')
OUTPUT = Path(r'Z:/projects/sequence_squad/revision_data/organised_data/probe_power_spectra//') 

def get_record_node_path_list(root_folder):
    xml_paths = []
    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check for any .xml files in the current directory
        xml_files = [f for f in filenames if f.endswith('.xml')]
        # If there are .xml files, add the directory path to the list for each file
        for xml_file in xml_files:
            xml_paths.append(dirpath)
    return xml_paths

def get_record_node_path(root_folder):
    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check if 'settings.xml' is in the current directory
        if 'settings.xml' in filenames:
            return dirpath
    return None

def fscale(ns, si=1, one_sided=False):
    """
    Stole this code from the IBL

    [https://github.com/int-brain-lab/ibl-neuropixel/blob/main/src/ibldsp/fourier.py]

    numpy.fft.fftfreq returns Nyquist as a negative frequency so we propose this instead

    :param ns: number of samples
    :param si: sampling interval in seconds
    :param one_sided: if True, returns only positive frequencies
    :return: fscale: numpy vector containing frequencies in Hertz
    """
    fsc = np.arange(0, np.floor(ns / 2) + 1) / ns / si  # sample the frequency scale
    if one_sided:
        return fsc
    else:
        return np.concatenate((fsc, -fsc[slice(-2 + (ns % 2), 0, -1)]), axis=0)
    
class probe_mapper():

    '''
    Used to generate the power spectrum per channel
     during a particular session. Choice between one_shank (the first)
     or four shanks. 
    '''

    def __init__(self, mouse, session, dir_path,mousepath, mode = 'four_shanks', fourier_mode = 'welch', segment = 0):

        self.mouse = mouse
        self.session  = session
        self.mode = mode
        self.low_band = 1
        self.fourier_mode = fourier_mode
        self.segment = segment


        self.node_path = dir_path
        print(self.node_path)

        self.output_path =  mousepath 


        self.output_path.mkdir(parents=True, exist_ok=True)
        #reading
        recording = se.read_openephys(self.node_path, stream_id  = '1', block_index = self.segment)

        if recording.get_num_segments() > 1:
            recording = recording.select_segments(0)
            print('More than one segment. Selecting the first.')

        print(f'Reading recording at {self.node_path}')
        print(recording)

        if self.mode == 'one_shank':

            split_recording_dict = recording.split_by("group")

            print('Separating probe 1')
            self.probe1 = split_recording_dict[1]

        elif self.mode == 'four_shanks':

            self.probe1 = recording

        self.samp = self.probe1.sampling_frequency
        print(f'Sampling freq is {self.samp}Hz')

        print(f'Filtering {self.low_band}Hz - {(self.samp/2)-1}Hz')
        self.probe1 = si.bandpass_filter(recording=self.probe1, freq_min=self.low_band, freq_max=(self.samp/2)-1)

        if self.fourier_mode == 'multitaper':
            #Keep 10s of data
            print('Extracting 10s')
            self.traces =  (self.probe1.get_traces(start_frame=5*self.samp, end_frame=15*self.samp)).T

        elif self.fourier_mode == 'welch':
            #Keep 10s of data
            print('Extracting 10s')
            self.traces =  (self.probe1.get_traces(start_frame=5*self.samp, end_frame=15*self.samp)).T

        self.nChans, self.nSamps = self.traces.shape
        print('Data has %d channels and %d samples',(self.nChans,self.nSamps))

    def plot_10s_traces(self):

        print('Plotting traces')

        rec1 = si.bandpass_filter(recording=self.probe1, freq_min=300, freq_max=6000)
        rec = si.common_reference(rec1, operator="median", reference="global")

        # Plot with spikeinterface or sw.plot_traces
        w_ts = sw.plot_traces(rec, mode="map", time_range=(5, 15), show_channel_ids=True, order_channel_by_depth=True)

        # If w_ts is an Axes object, this will get the parent figure
        fig = w_ts.figure
        ax = w_ts.ax

        # Set the figure size (width, height)
        fig.set_size_inches(10, 15)

        # Get the current y-ticks
        yticks = ax.get_yticks()

        ax.set_xlabel('time (s)')

        # Set the y-ticks to show only every 10th channel
        new_yticks = yticks[::10]
        ax.set_yticks(new_yticks)

        fig.suptitle('10s of bandpassed (300-6k Hz), CMR recording')

        fig.savefig(self.output_path / 'traces.png')

        plt.close()

    def fourier(self):
        '''
        Calculates the Fourier transform of the first channel. Can use multitaper
        (precise w small sample sizes, but slow), or fft (fast, maybe  imprecise in
        low frequencies)

        Returns:
            pxx: power spectrum in V**2
            f: frequencies of pxx in Hz
        '''
        if  self.fourier_mode  == 'multitaper':

            psd = multitaper.MTSpec(x=self.traces[0,:]/10E6, dt=1.0/self.samp, nw=5) # run the multitaper spectrum
            self.pxx, self.f = psd.spec, psd.freq # unpack power spectrum and frequency from output
            plot_range = (self.f<=10) & (self.f>=0) # find the frequencies we want to plot
            fig, ax = plt.subplots()

            # Correct method call for semilogy
            ax.semilogy(self.f[plot_range], self.pxx[plot_range])

            # Set axis labels
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power (V**2)')


            fig.suptitle('Multitaper spectrum of first channel')
            fig.savefig(self.output_path / 'spectrum.png')

            plt.close()


    def probe_spectrum(self):

        self.pxx_list = list(np.zeros(len(self.probe1.channel_ids)))
        self.f_list = list(np.zeros(len(self.probe1.channel_ids)))

        if self.fourier_mode ==  'multitaper':

            for i in tqdm(np.arange(len(self.pxx_list))):
                print (i)
                psd = multitaper.MTSpec(x=self.traces[i,:]/10E6, dt=1.0/self.samp, nw=5) # run the multitaper spectrum
                pxx, f = psd.spec, psd.freq # unpack power spectrum and frequency from output
                self.pxx_list[i] = pxx
                self.f_list[i] = f

            freq_per_channel = {
                'channel': np.arange(len(self.probe1.channel_ids)), 
                'pxx': self.pxx_list, 
                'f': self.f_list
            }

            self.freq =  pd.DataFrame(freq_per_channel)

        if self.fourier_mode == 'welch':

            windows = self.make_windows(4)

            self.f_list = [fscale(self.window_samples, 1/self.samp, one_sided=True) for _ in range(self.nChans)]

            spectra = np.zeros((self.nChans, len(self.f_list[0])))

            for window in tqdm(np.arange(4)):
                start, end = int(windows[window, 0]), int(windows[window, 1])
                trace = self.traces[:, start:end]
                _, w = signal.welch(
                trace/10E6, fs=self.samp, window='hann', nperseg=self.window_samples,
                detrend='constant', return_onesided=True, scaling='density', axis=-1
                )

                spectra += w

            spectrum = spectra/4

            self.pxx_list = spectrum.tolist()

            freq_per_channel = {
                'channel': np.arange(len(self.probe1.channel_ids)), 
                'pxx': self.pxx_list,
                'f': self.f_list
            }

            self.freq =  pd.DataFrame(freq_per_channel)

    def make_windows(self, n_windows = 4):  
        self.window_samples = self.nSamps//n_windows
        print(f'Window samples are {self.window_samples} samples')
        windows = np.zeros((n_windows, 2))
        index = 0
        for window in np.arange(n_windows):
            windows[index, 0] = index*self.window_samples
            windows[index, 1] = index*self.window_samples + self.window_samples
            index +=1

        return windows
    

    def get_delta_power(self, pxx, f):

        '''
        Extracts the power in the delta band and transforms it into
        decibels. 
        '''
        # Convert the frequency list to a numpy array so you can use comparisons

        f = np.array(f)
        pxx = np.array(pxx)

        # Define the frequency range of interest (0-4 Hz)
        band_range = (f >= 0) & (f <= 4)

        # Calculate the total power in the 0-4 Hz band by summing the power values in that range
        power_band = np.sum(pxx[band_range])

        # Convert the power to dB
        power_db = 10 * np.log10(power_band)

        return power_db
    
    def calculate_delta_power(self):

        # ISSUE WITH THE SAPE OF FREQ: SHOULD BE CONSISTENT; A LIST OF SPECTRA

        self.freq['delta_power'] =  [self.get_delta_power(pxx, f) for pxx,f in zip(self.pxx_list, self.f_list)]
        self.freq.to_csv(self.output_path / 'freq.csv')

    def build_probemap(self):

        '''
        Assuming the order of the  channels is the order  of the  
        contact points, which I took from the openephys code
        '''
        self.probemap = self.probe1.get_probe().to_dataframe()

        self.probemap['channel'] = self.probe1.channel_ids
        self.probemap['dbs'] = self.freq['delta_power']

        fig, ax = plt.subplots()

        # Create a scatter plot
        sc = ax.scatter(self.probemap['x'], self.probemap['y'], c=self.probemap['dbs'], cmap='viridis', s=50)

        # Add color bar for the 'dfs' values (make sure to pass the scatter plot object `sc`)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Delta power (power in Db from 0 to 4 Hz in a signal in V)')

        if self.mode == 'one_shank':

            # Set x-axis limits
            ax.set_xlim((100, 450))
            ax.set_ylabel('depth (microns)')

            fig.suptitle('Map of the shank 1 and delta power')

        elif self.mode == 'four_shanks':

            # Set x-axis limits
            ax.set_ylabel('depth (microns)')

            fig.suptitle('Map of the four shanks and delta power')
        
        #save
        fig.savefig(self.output_path / 'delta_map.png')
        self.probemap.to_csv(self.output_path / 'probemap.csv')

        plt.close()

class  whole_probe():

    '''
    Collects data generated by probe_mapper to generate a map of delta power
    over the whole probe. 
    '''

    def __init__(self, mouse,OUTPUT, mode = 'four_shanks'):

        self.mouse  = mouse
        self.mode = mode

        self.probemap = pd.DataFrame()

        self.root_path = OUTPUT  

        mousepath = OUTPUT
        mousepath.mkdir(exist_ok=True)

        if self.mode == 'one_shank':

            self.output_path =  mousepath / 'whole_probe'
            self.output_path.mkdir(exist_ok=True)
        
        elif self.mode == 'four_shanks':

            self.output_path =  mousepath / 'whole_probe_four_shanks'
            self.output_path.mkdir(exist_ok=True)

    def build_whole_probemap(self):

        for root, dirs, files in os.walk(self.root_path):
            for directory in dirs:

                self.session = directory

                dir_path = os.path.join(root, directory)

                if not dir_path.endswith(self.mode):

                    self.colect_data(dir_path)
        
        self.probemap.to_csv(self.output_path / 'complete_probemap.csv')

    def colect_data(self, path):

        # Construct the full path to the CSV file
        probemap_path = os.path.join(path, 'probemap.csv')

        # Check if the file exists
        if os.path.exists(probemap_path):
            # Read the CSV file
            new_probemap = pd.read_csv(probemap_path)

            new_probemap['session'] = self.session

            # Check if the class has the probemap attribute
            if hasattr(self, 'probemap'):
                # Append the new data to the existing probemap DataFrame
                self.probemap = pd.concat([self.probemap, new_probemap], ignore_index=True)
            else:
                # If probemap does not exist, assign the new CSV as probemap
                self.probemap = new_probemap
        else:
            print(f"{probemap_path} does not exist.")

    def process_probemap(self):
            '''
            Deals with  duplicate entries for a  single  point  (more than two
            sessions on the  same bank) by keeping the average value for  dbs
            on that point in space.  )
            '''
            self.processed_probemap = self.probemap.groupby(['x', 'y', 'contact_ids'], as_index=False)['dbs'].mean()
            self.processed_probemap.to_csv(self.output_path / 'processed_probemap.csv')

            
    def plot_probemap(self, combine='processed'):
        '''
        Combine stands for whether all the datapoints are plotted, or
        datapoints of the same channel are averaged across sessions.
        '''

        fig, ax = plt.subplots()

        y_range = max(self.probemap['y'])-min(self.probemap['y'])

        fig.set_figheight(5+(0.0078*y_range)) #scale figure size with y range, this seems like a sensible scaling
        fig.set_figwidth(10)

        # Create a scatter plot
        if combine == 'raw':
            sc = ax.scatter(self.probemap['x'], self.probemap['y'], c=self.probemap['dbs'], cmap='viridis', s=50)
            # Add contact IDs for 'raw'
            for i in range(len(self.probemap)):
                ax.annotate(self.probemap['contact_ids'].iloc[i], (self.probemap['x'].iloc[i], self.probemap['y'].iloc[i]),
                            fontsize=6, color='black', ha='center', va='center')

        elif combine == 'processed':
            sc = ax.scatter(self.processed_probemap['x'], self.processed_probemap['y'], c=self.processed_probemap['dbs'], cmap='viridis', s=50)
            # Add contact IDs for 'processed'
            for i in range(len(self.processed_probemap)):
                ax.annotate(self.processed_probemap['contact_ids'].iloc[i], 
                            (self.processed_probemap['x'].iloc[i]+60, self.processed_probemap['y'].iloc[i]),
                            fontsize=6, color='black', ha='center', va='center')

        # Add color bar for the 'dfs' values
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Delta power (power in Db from 0 to 4 Hz in a signal in V)')

        # Set y-ticks to have more values
        all_y_ticks = np.linspace(self.probemap['y'].min(), self.probemap['y'].max(), num=50)  # Adjust num for more/less ticks
        ax.set_yticks(all_y_ticks)

        if self.mode == 'one_shank':
            # Set x-axis limits
            ax.set_xlim((100, 450))
            ax.set_ylabel('depth (microns)')
            ax.set_xlabel('microns')
            fig.suptitle('Map of the shank 1 and delta power')

        elif self.mode == 'four_shanks':
            # Set x-axis limits
            ax.set_ylabel('depth (microns)')
            ax.set_xlabel('microns')
            fig.suptitle('Map of the four shanks and delta power')

        # Save the figure
        fig.savefig(self.output_path / 'complete_delta_map.png')
