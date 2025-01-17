## Import

import os
import scipy.io
import numpy as np
import os, importlib
import matplotlib.pyplot as plt
import statistics
import scipy.stats
import matplotlib.patches as mpatches
import tkinter as tk
import pandas as pd
import pickle
from open_ephys.analysis import Session
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import glob
import cv2
from tqdm import tqdm
import json

## Functions

def diff(data):
    diff = []
    for i in range(1,len(data)):
        diff = diff + [data[i] - data[i-1]]
    return(diff)

def create_spike_time_vectors(spike_times,clusters):
    spiketimevectors = []
    for i in np.unique(clusters):
        spiketimevectors = spiketimevectors + [spike_times[np.where(clusters==i)[0]]]
    return spiketimevectors

def SaveFig(file_name,figure_dir):
    if os.path.isfile(figure_dir + file_name ):
        os.remove(figure_dir + file_name)  
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(figure_dir + file_name)
    plt.close()
    

def Reformat_PokeEphysTS(PokeIn_EphysTS):
    P1_PokeIn_EphysTS = np.array([])
    P2_PokeIn_EphysTS = np.array([])
    for ind, item in enumerate(PokeIn_EphysTS):
        if ind > 0:
            P1_PokeIn_EphysTS = np.append(P1_PokeIn_EphysTS,[PokeIn_EphysTS[ind-1]])
            
            if item == 'NaN':
                P2_PokeIn_EphysTS = np.append(P2_PokeIn_EphysTS,[np.nan])
            else:
                P2_PokeIn_EphysTS =  np.append(P2_PokeIn_EphysTS,[item])
    return np.array(P1_PokeIn_EphysTS),P2_PokeIn_EphysTS



## new version for new open ephys tools 
def align_open_ephys_processors(main_processor_tuple, aux_processor_tuples,raw_data_directory, sync_channel=1):

    session_data = Session(str(raw_data_directory))
    if len(session_data.recordnodes) != 1:
        raise ValueError("should be exactly one record node.")
    if len(session_data.recordnodes[0].recordings) != 1:
        raise ValueError("Should be exactly one recording.")
    for rn, recordnode in enumerate(session_data.recordnodes):
        for r, recording in enumerate(recordnode.recordings):
            # Synch
            recording.add_sync_line(
                sync_channel,
                main_processor_tuple[0],
                main_processor_tuple[1],
                main=True,
            )
            for aux_processor in aux_processor_tuples:
                recording.add_sync_line(
                    sync_channel,
                    aux_processor[0],
                    aux_processor[1],
                    main=False,
                )
            print('this should be zero:')
            print(rn)
        
    return recording


def sequence_contains_sequence(haystack_seq, needle_seq, string):
    start_index = []
    for i in range(0, len(haystack_seq) - len(needle_seq) + 1):
        if needle_seq == haystack_seq[i:i+len(needle_seq)]:
            start_index = start_index + [i]
            print(string + ' barcode found')
    return start_index

def find_first_poke_times(trial_id,ports, poke_in_array):

    first_poke_index = []
    count = 1

    for index,item in enumerate(trial_id):
        if item == count:
            if ports[index] == 2:
                count = count + 1
                first_poke_index = first_poke_index + [index]
            
    return poke_in_array[np.array(first_poke_index)]

def align_firstpoke_camera_timestamps(trial_id,Trial_start_ts,All_Port_references_sorted):
    counter = 0
    trial_ts_aligned = []
    for index,item in enumerate(trial_id):
        if All_Port_references_sorted[index] == 2.0:
            if item > counter:
                counter = counter + 1
                #getting an error 05/01/22 -> I think if the next trial started (TTL went high) but no poke was recorded then there wont be a poke timestamp so account for this: 
                if not len(Trial_start_ts) == counter - 1:
                    trial_ts_aligned = trial_ts_aligned + [Trial_start_ts[counter-1]]
                else:
                    trial_ts_aligned = trial_ts_aligned + ['NaN']
            else:
                trial_ts_aligned = trial_ts_aligned + ['NaN']
        else:
            trial_ts_aligned = trial_ts_aligned + ['NaN']
    return trial_ts_aligned

def align_trial_start_end_timestamps(trial_id,Trial_start_ts):
    trial_ts_aligned = []
    counter = 1
    for index, item in enumerate(trial_id):
        if counter ==  item:
            trial_ts_aligned = trial_ts_aligned + [Trial_start_ts[counter-1]]
        else:
            counter = counter + 1
            trial_ts_aligned = trial_ts_aligned + [Trial_start_ts[counter-1]]
    
    return trial_ts_aligned
            

def clock_find_first_poke_times(trial_id,ports, poke_in_array,trial_seqs):

    first_poke_index = []
    count = 1

    for index,item in enumerate(trial_id):
        if item == count:
            if ports[index] == trial_seqs[index][0]:
                count = count + 1
                first_poke_index = first_poke_index + [index]
            
    return poke_in_array[np.array(first_poke_index)]

def clock_align_firstpoke_camera_timestamps(trial_id,Trial_start_ts,All_Port_references_sorted,trial_seqs):
    counter = 0
    trial_ts_aligned = []
    for index,item in enumerate(trial_id):
        if All_Port_references_sorted[index] == trial_seqs[index][0]:
            if item > counter:
                counter = counter + 1
                #getting an error 05/01/22 -> I think if the next trial started (TTL went high) but no poke was recorded then there wont be a poke timestamp so account for this: 
                if not len(Trial_start_ts) == counter - 1:
                    trial_ts_aligned = trial_ts_aligned + [Trial_start_ts[counter-1]]
                else:
                    trial_ts_aligned = trial_ts_aligned + ['NaN']
            else:
                trial_ts_aligned = trial_ts_aligned + ['NaN']
        else:
            trial_ts_aligned = trial_ts_aligned + ['NaN']
    return trial_ts_aligned

def AlignToTriggersAndFIndEphysTimestamps(Port_intimes,trial_id,first_poke_times,trial_start,TrialStart_EphysTime,FirstPoke_EphysTime):

    new_TS = []
    for index, trial in enumerate(trial_id):
        if np.isnan(Port_intimes[index]):
            new_TS = new_TS + [np.nan]
        else:

            current_poke_event_time = Port_intimes[index]

            # find ech relevant timestamps
            CurrentTrial_startTS = trial_start[trial-1]
            First_pokeTS = first_poke_times[trial-1]

            # last trial has no next trial start
            if trial == trial_id[-1]:
                NextTrial_startTS = 9999999999999
            else:
                NextTrial_startTS = np.unique(trial_start)[trial]

            # find the ts current poke event is closest to
            trialstart_diff =  abs(CurrentTrial_startTS - current_poke_event_time)

            EphysTS = TrialStart_EphysTime[trial-1]
            current_dist = current_poke_event_time - CurrentTrial_startTS 
            distance = EphysTS + current_dist

            new_TS = new_TS + [distance]
            
    return(new_TS)

def Determine_Transition_Times_and_Types(All_PortIn_Times_sorted ,All_PortOut_Times_sorted, All_Port_references_sorted):
    out_in= []
    in_in = []
    transition_type = []
    out_in_transition_reference = []
    in_in_transition_reference = []
    for index,port in enumerate(All_Port_references_sorted):
        if index > 0:
            out_in = out_in + [All_PortIn_Times_sorted[index] - All_PortOut_Times_sorted[index-1] ]
            out_in_transition_reference = out_in_transition_reference + [All_PortOut_Times_sorted[index-1]]

            in_in = in_in + [All_PortIn_Times_sorted[index] - All_PortIn_Times_sorted[index-1] ]
            in_in_transition_reference = in_in_transition_reference + [All_PortIn_Times_sorted[index-1]]

            transition_type = transition_type + [int(str(All_Port_references_sorted[index-1]) + str(port))]

    return (np.array(out_in),np.array(in_in) ,np.array(transition_type),out_in_transition_reference,in_in_transition_reference)

def Determine_transition_matrix(prev_port,current_port):
    Transition = (prev_port * 10) + current_port
    return Transition

def Start_End_port_id(Transition_types,start_end_arg):
    output = []
    for item in Transition_types:
        String = str(item)
        output = output + [int(String[start_end_arg])]
    return output

def determine_RepeatPort_events(start_port_ids,end_port_ids):
    Port_repeat = []
    for index, item in enumerate(start_port_ids):
        if item == end_port_ids[index]:
            Port_repeat = Port_repeat + [0]
        else: 
            Port_repeat = Port_repeat + [1]
    return Port_repeat    

def filter_transitons_by_latency(Transition_times, Upper_Filter):
    Filtered_transitions = []
    for item in Transition_times:
        if item < Upper_Filter:
            Filtered_transitions = Filtered_transitions + [1]
        else:
            Filtered_transitions = Filtered_transitions + [0]
    return Filtered_transitions

def find_files(filename, search_path):
    result = []

    #Walking top-down from the root
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))

    return result

def find_folder_path(parent_folder, target_folder):
    for root, dirs, files in os.walk(parent_folder):
        if target_folder in dirs:
            return os.path.join(root, target_folder)
        # If the target folder is not found
    return (print('not found'))

def test_timestamps_(Trial_start_ts,Trial_start_Camera_Ts):
    working = False
    try:
        trial_start_difference = (np.diff(Trial_start_ts) - np.diff(Trial_start_Camera_Ts)[0:-1])
        working = True

    except:
        print('broken - being fixed')
        tester = False
        while tester == False:
            Trial_start_Camera_Ts,tester = fix_missing_triggers(Trial_start_ts,Trial_start_Camera_Ts)
            print('fixed')
        print('all fixed')
    try:
        trial_start_difference = (np.diff(Trial_start_ts) - np.diff(Trial_start_Camera_Ts)[0:-1])
        working = True
        print('working')
    except:
        print('fail')

    for item in (trial_start_difference):
        if abs(item) > 1:
            raise NameError('Test not passed, Timestamps dont line up!')

    print('test passed!')
    
    Trial_start_Camera_Ts= Trial_start_Camera_Ts
    
    return Trial_start_Camera_Ts

def fix_missing_triggers(Trial_start_ts,Trial_start_Camera_Ts):

    fixed_trial_strart_ts= []
    for index,item in enumerate(np.diff(Trial_start_ts)):
        if abs(item - np.diff(Trial_start_Camera_Ts)[index]) > 1:
#             missing_trigger = Trial_start_Camera_Ts[index]+np.diff(Trial_start_ts)[index]
            missing_trigger = np.nan
            fixed_trial_strart_ts = np.insert(Trial_start_Camera_Ts,index+1,[missing_trigger])
            print(index)
            break
    
    try:
        trial_start_difference = (np.diff(Trial_start_ts) - np.diff(fixed_trial_strart_ts)[0:-1])
        for item in (trial_start_difference):
            if abs(item) > 1:
                tester = False
            else:
                tester = True
    except:
        tester = False

    return fixed_trial_strart_ts, tester


def align_to_start_ts(trials,fixed_cam_ts):
    Camera_timestamps = []
    Camera_time = []
    for trial in trials :
        Camera_timestamps.append(fixed_cam_ts[trial-1])
        Camera_time.append(fixed_cam_ts[trial-1])
    return Camera_timestamps,Camera_time

## alignment: 
def align_allpokes_to_cam_trialstart(Trials,trial_start_bpod_ts,PokeIn_Time,Fixed_back_cam_tstart_ts):

    cam_poke_times = []
    for index,trial in enumerate(Trials):
        bpod_trial_start = trial_start_bpod_ts[trial-1]
        poke_time = PokeIn_Time[index]
        diff = poke_time - bpod_trial_start
        cam_poke_times = cam_poke_times + [(Fixed_back_cam_tstart_ts[trial-1]) + diff]

    return cam_poke_times

def Determine_Transition_Times_and_Types(All_PortIn_Times_sorted ,All_PortOut_Times_sorted, All_Port_references_sorted):
    out_in= []
    in_in = []
    transition_type = []
    out_in_transition_reference = []
    in_in_transition_reference = []
    for index,port in enumerate(All_Port_references_sorted):
        if index > 0:
            out_in = out_in + [All_PortIn_Times_sorted[index] - All_PortOut_Times_sorted[index-1] ]
            out_in_transition_reference = out_in_transition_reference + [All_PortOut_Times_sorted[index-1]]

            in_in = in_in + [All_PortIn_Times_sorted[index] - All_PortIn_Times_sorted[index-1] ]
            in_in_transition_reference = in_in_transition_reference + [All_PortIn_Times_sorted[index-1]]

            transition_type = transition_type + [int(str(All_Port_references_sorted[index-1]) + str(port))]

    return (np.array(out_in),np.array(in_in) ,np.array(transition_type),out_in_transition_reference,in_in_transition_reference)


def Start_End_port_id(Transition_types,start_end_arg):
    output = []
    for item in Transition_types:
        String = str(item)
        output = output + [int(String[start_end_arg])]
    return output

def determine_RepeatPort_events(start_port_ids,end_port_ids):
    Port_repeat = []
    for index, item in enumerate(start_port_ids):
        if item == end_port_ids[index]:
            Port_repeat = Port_repeat + [0]
        else: 
            Port_repeat = Port_repeat + [1]
    return Port_repeat    

def filter_transitons_by_latency(Transition_times, Upper_Filter):
    Filtered_transitions = []
    for item in Transition_times:
        if item < Upper_Filter:
            Filtered_transitions = Filtered_transitions + [1]
        else:
            Filtered_transitions = Filtered_transitions + [0]
    return Filtered_transitions

def load_H5_bodypart_ports(tracking_path):

    back_file = pd.read_hdf(tracking_path)

    # drag data out of the df
    scorer = back_file.columns.tolist()[0][0]
    
    # drag data out of the df
    scorer = back_file.columns.tolist()[0][0]

    # list all the columns in the dataframe 
    columns = back_file[scorer].columns.get_level_values(0).unique()

    bodypart= [[]]*len(columns)
    for index, column in enumerate(columns):
        bodypart[index] =back_file[scorer][column]

    for item in bodypart:
        clean_and_interpolate(item,0.9)
        
    return bodypart


def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def files(folderpath, pattern="*"):
    """
        returns all files folders in a given folder matching a pattern
    """
    return [f for f in folderpath.glob(pattern) if f.is_file()]

def clean_and_interpolate(head_centre,threshold):
    bad_confidence_inds = np.where(head_centre.likelihood.values<threshold)[0]
    newx = head_centre.x.values
    newx[bad_confidence_inds] = 0
    newy = head_centre.y.values
    newy[bad_confidence_inds] = 0

    start_value_cleanup(newx)
    interped_x = interp_0_coords(newx)

    start_value_cleanup(newy)
    interped_y = interp_0_coords(newy)
    
    head_centre['interped_x'] = interped_x
    head_centre['interped_y'] = interped_y
    
def start_value_cleanup(coords):
    # This is for when the starting value of the coords == 0; interpolation will not work on these coords until the first 0 
    #is changed. The 0 value is changed to the first non-zero value in the coords lists
    for index, value in enumerate(coords):
        working = 0
        if value > 0:
            start_value = value
            start_index = index
            working = 1
            break
    if working == 1:
        for x in range(start_index):
            coords[x] = start_value
            
def interp_0_coords(coords_list):
    #coords_list is one if the outputs of the get_x_y_data = a list of co-ordinate points
    for index, value in enumerate(coords_list):
        if value == 0:
            if coords_list[index-1] > 0:
                value_before = coords_list[index-1]
                interp_start_index = index-1
                #print('interp_start_index: ', interp_start_index)
                #print('interp_start_value: ', value_before)
                #print('')

        if index < len(coords_list)-1:
            if value ==0:
                if coords_list[index+1] > 0:
                    interp_end_index = index+1
                    value_after = coords_list[index+1]
                    #print('interp_end_index: ', interp_end_index)
                    #print('interp_end_value: ', value_after)
                    #print('')

                    #now code to interpolate over the values
                    try:
                        interp_diff_index = interp_end_index - interp_start_index
                    except UnboundLocalError:
#                         print('the first value in list is 0, use the function start_value_cleanup to fix')
                        break
                    #print('interp_diff_index is:', interp_diff_index)

                    new_values = np.linspace(value_before, value_after, interp_diff_index)
                    #print(new_values)

                    interp_index = interp_start_index+1
                    for x in range(interp_diff_index):
                        #print('interp_index is:', interp_index)
                        #print('new_value should be:', new_values[x])
                        coords_list[interp_index] = new_values[x]
                        interp_index +=1
        if index == len(coords_list)-1:
            if value ==0:
                for x in range(30):
                    coords_list[index-x] = coords_list[index-30]
                    #print('')
#     print('function exiting')
    return(coords_list)


def extract_frames_by_number(input_file, output_file, start_frame, end_frame):
    """
    Extracts frames from a video file based on frame numbers and saves as a new video.

    :param input_file: Path to the input video file.
    :param output_file: Path to save the output video file.
    :param start_frame: The starting frame number (inclusive).
    :param end_frame: The ending frame number (inclusive).
    """
    # Open the input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output file

    # Open the VideoWriter
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Read and process frames
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Write frames in the specified range
        if start_frame <= frame_number <= end_frame:
            out.write(frame)

        # Stop if end frame is reached
        if frame_number > end_frame:
            break

        frame_number += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Extracted video saved to {output_file}")
    
    
def align_to_start_ts(trials,fixed_cam_ts):
    Camera_timestamps = []
    Camera_time = []
    for trial in trials :
        Camera_time.append(fixed_cam_ts[trial-1])
    return Camera_time

## alignment: 
def align_allpokes_to_cam_trialstart(Trials,trial_start_bpod_ts,PokeIn_Time,Fixed_back_cam_tstart_ts):

    cam_poke_times = []
    for index,trial in enumerate(Trials):
        bpod_trial_start = trial_start_bpod_ts[trial-1]
        poke_time = PokeIn_Time[index]
        diff = poke_time - bpod_trial_start
        cam_poke_times += [Fixed_back_cam_tstart_ts[trial-1] + diff]

    return cam_poke_times

def find_closest_cameraframe_index(behav_cam_ts, back_aligned_time):
    closest_inds = []
    for index in tqdm(range(len(back_aligned_time))):
        time = back_aligned_time[index]
        # find the ind where the camera timestamp is closest to the poke in time
        closest_inds += [behav_cam_ts.index[0] + np.argmin(np.abs(behav_cam_ts['Time Stamps'].values - time))]
    return closest_inds



def process_probe_data_bool(organised_ephys_path,aux_processor_tuples):
    process = False
    if not 'global-timstamps_event-df.pkl' in os.listdir(organised_ephys_path):
        if not 'main_continuous_global_ts_probeA.npy' in os.listdir(organised_ephys_path):
            if not 'LFP' in aux_processor_tuples[0][-1]:
                if not 'main_continuous_global_ts_probeA_LFP.npy' in os.listdir(organised_ephys_path):
                    process = True
            else:
                if not 'main_continuous_global_ts_probeB.npy' in os.listdir(organised_ephys_path):
                    process = True
    return process



def align_camera_to_ephys_ts(camera_times_s, where_ttl_changes, cam_trigger_times, ephys_times_synced_A):
    """
    Aligns camera timestamps to electrophysiology (ephys) timestamps.

    Args:
        camera_times_s (list): Camera timestamps in seconds.
        where_ttl_changes (list): Indices indicating where TTL changes occur.
        cam_trigger_times (list): Trigger times for the camera.
        ephys_times_synced_A (list): Synchronized ephys timestamps.

    Returns:
        list: Aligned ephys timestamps corresponding to the camera timestamps.
    """
    aligned_time = []  # List to store aligned timestamps
    count = 0  # Counter for TTL changes

    for index in tqdm(range(len(camera_times_s))):
        if count < len(where_ttl_changes):
            if index < where_ttl_changes[count]:
                # Calculate the time difference
                diff = cam_trigger_times[count] - camera_times_s[index]
                aligned_time.append(ephys_times_synced_A[count] - diff)
                if diff < 0:
                    print('Error: Negative time difference encountered!')
            else:
                # Use the current ephys time when index exceeds TTL change boundary
                aligned_time.append(ephys_times_synced_A[count])
                count += 1
        else:
            # Handle timestamps beyond the last TTL change
            diff = camera_times_s[index] - cam_trigger_times[count - 1]
            aligned_time.append(ephys_times_synced_A[count - 1] + diff)
            if diff < 0:
                print('Error: Negative time difference encountered!')

    return aligned_time

def create_clusters_with_depths(spikes_path,good_indices,spike_times,clusters):
    
    spike_templates = np.squeeze(np.load(spikes_path + 'spike_templates.npy'))
    temps = np.load(spikes_path + 'templates.npy')
    winv = np.load(spikes_path + 'whitening_mat_inv.npy')
    y_coords = np.squeeze(np.load(spikes_path + 'channel_positions.npy'))[:,1]

    real_spikes = spike_times[good_indices]
    real_clusters = clusters[good_indices]
    real_spike_templates = spike_templates[good_indices]

    # find how many spikes per cluster and then order spikes by which cluster they are in
    counts_per_cluster = np.bincount(real_clusters)

    sort_idx = np.argsort(real_clusters)
    sorted_clusters = real_clusters[sort_idx]
    sorted_spikes = real_spikes[sort_idx]
    sorted_spike_templates = real_spike_templates[sort_idx]

    # find depth for each spike
    # this is translated from Cortex Lab's MATLAB code
    # for more details, check out the original code here:
    # https://github.com/cortex-lab/spikes/blob/master/analysis/templatePositionsAmplitudes.m

    temps_unw = np.zeros(temps.shape)
    for t in range(temps.shape[0]):
        temps_unw[t, :, :] = np.dot(temps[t, :, :], winv)

    temp_chan_amps = np.ptp(temps_unw, axis=1)
    temps_amps = np.max(temp_chan_amps, axis=1)
    thresh_vals = temps_amps * 0.3

    thresh_vals = [thresh_vals for i in range(temp_chan_amps.shape[1])]
    thresh_vals = np.stack(thresh_vals, axis=1)

    temp_chan_amps[temp_chan_amps < thresh_vals] = 0

    y_coords = np.reshape(y_coords, (y_coords.shape[0], 1))
    temp_depths = np.sum(
        np.dot(temp_chan_amps, y_coords), axis=1) / (np.sum(temp_chan_amps,axis=1))

    sorted_spike_depths = temp_depths[sorted_spike_templates]

    # create neurons and find region
    out_clusters= []
    spike_vectors = []
    cluster_depths = []

    accumulator = 0

    for idx, count in enumerate(counts_per_cluster):
        if count > 0:
            new_spike_times = list(np.sort(sorted_spikes[accumulator:accumulator + count]))
            cluster_depth = np.mean(sorted_spike_depths[accumulator:accumulator + count])
            spike_vectors = spike_vectors + [new_spike_times]
            cluster_depths = cluster_depths + [cluster_depth]
            out_clusters = out_clusters + [idx]
            accumulator += count

    # make dataframe:
    good_df = pd.DataFrame(
        {'cluster_id' : out_clusters,
        'cluster_depth' : cluster_depths,
            'Spike_times':spike_vectors})
    
    return good_df

def check_if_probe_is_flipped(A_probes):

    # Check if 'CP' appears before 'ccb or ccg' in the list of region acronyms, it shouldnt if the probe is the right way up
    if 'CP' in A_probes['Region acronym'].values and 'ccb' in A_probes['Region acronym'].values:
        if list(A_probes['Region acronym'].values).index('CP') < list(A_probes['Region acronym'].values).index('ccb'):
            print('cp appears first ')
            flipped = True
        else:
            print('cp appears second - good')
            flipped = False
    elif 'CP' in A_probes['Region acronym'].values and 'ccg' in A_probes['Region acronym'].values:
        if list(A_probes['Region acronym'].values).index('CP') < list(A_probes['Region acronym'].values).index('ccg'):
            print('cp appears first ')
            flipped = True
        else:
            print('cp appears second - good')
            flipped = False
    else:
        print('error')
    return flipped

def check_if_probeB_is_flipped(B_probes):

    # Check if 'CP' appears before 'ccb or ccg' in the list of region acronyms, it shouldnt if the probe is the right way up
    if 'LP' in B_probes['Region acronym'].values and 'cing' in B_probes['Region acronym'].values:
        if list(B_probes['Region acronym'].values).index('LP') < list(B_probes['Region acronym'].values).index('cing'):
            print('cing appears first ')
            flipped = True
        else:
            print('LP appears second - good')
            flipped = False
    elif 'LP' in B_probes['Region acronym'].values and 'ccs' in B_probes['Region acronym'].values:
        if list(B_probes['Region acronym'].values).index('LP') < list(B_probes['Region acronym'].values).index('ccs'):
            print('ccs appears first ')
            flipped = True
        else:
            print('LP appears second - good')
            flipped = False
    else:
        print('error')
    return flipped