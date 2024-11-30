import scipy.io
import os, importlib
import matplotlib.pyplot as plt
import statistics
import scipy.stats
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
import tkinter as tk
from matplotlib import gridspec
import matplotlib.colors
import numpy as np
import pandas as pd
import os.path, time
import pickle



def Import_Bpod_DataFiles(InputPath):
    
    '''
    Load in all '.mat' files for a given folder and convert them to python format:
    '''
    Behav_Path = sorted(os.listdir(InputPath))
    Behav_Data = {} #set up file dict
    File_dates = []
    Sessions = 0 # for naming each data set within the main dict

    for file in Behav_Path:
        if file[-2] == 'a': #if its a .mat and not a fig
            if os.stat(InputPath+ file).st_size > 200000: #more thann 200kb 
                if file != '.DS_Store': #if file is not the weird hidden file 
                    print(file)
                    Current_file = loadmat(InputPath + file)
                    Behav_Data[Sessions] = Current_file
                    Sessions = Sessions + 1
                    File_dates = File_dates + [file[-19:-4]]
    
    return Behav_Data, Sessions, Behav_Path,File_dates

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return check_keys(data)

def check_keys(temp_dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in temp_dict:
        if isinstance(temp_dict[key], scipy.io.matlab.mio5_params.mat_struct):
            temp_dict[key] = todict(temp_dict[key])
          
    return temp_dict        

def todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    temp_dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            temp_dict[strg] = todict(elem)
        else:
            temp_dict[strg] = elem
    return temp_dict

def convert_nested_structs(Behav_data):
    # Nested structures are loaded in strangly so use Francesca's function to extract the important ones:
    if 'RawEvents' in Behav_data['SessionData']: #dont process bad data
        Current_data = Behav_data['SessionData']['RawEvents']['Trial']
        for trial_num, trial in enumerate(Current_data,1):
            Current_data[trial_num-1] = todict(Current_data[trial_num-1])

# Timestamp preprocessing:

def load_camera_timestamps(InputPath):
    Camera_timestamps = pd.read_csv((InputPath), sep = ' ', header=None, names=['Trigger', 'Timestamp', 'blank'], index_col=2)
    del Camera_timestamps['blank']
    return Camera_timestamps 

def converttime(time):
    #offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds

def uncycle(time):
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128

def convert_uncycle_Timestamps(Camera_ts):
    ##################   Convert the timestamps into seconds and uncycle them:
    t_stamps = {}  
    stamps_s = []
    for indx, row in Camera_ts.iterrows():
        if row.Trigger > 0: 
            timestamp_new = converttime(Camera_ts.at[indx,'Timestamp'])
            stamps_s.append(timestamp_new)
        else:    
            raise ValueError('Timestamps are broken')
    t_stamps = uncycle(stamps_s)
    t_stamps = t_stamps - t_stamps[0] # make first timestamp 0 and the others relative to this 
    return(t_stamps)

def check_timestamps(t_stamps, Frame_rate):
    # plot 1/(diff between time stamps). This tells you roughly the frequency and if you've droppped frames.
    Frame_gaps = 1/np.diff(t_stamps)
    Frames_dropped = 0
    for gaps in Frame_gaps:
        if gaps < (Frame_rate-5) or gaps > (Frame_rate+5):
            Frames_dropped = Frames_dropped + 1
    print('Frames dropped = ' + str(Frames_dropped))

    plt.suptitle('Frame rate = ' + str(Frame_rate) + 'fps', color = 'red')
    frame_gaps = plt.hist(Frame_gaps, bins=100)
    plt.xlabel('Frequency')
    plt.ylabel('Number of frames')
    
def find_trigger_states(Camera_ts_raw):
    down_state = list(Camera_ts_raw.loc[:,'Trigger'])[0]
    down_state_times = np.where(Camera_ts_raw.loc[:,'Trigger'] == down_state)
    Triggers_temp = np.ones(len(Camera_ts_raw.loc[:,'Trigger']))
    for index in down_state_times:
        Triggers_temp[index] = 0
    trigger_state = Triggers_temp
    return trigger_state

# extract specific Bpod data values 

def extract_trial_timestamps(Behav_data):
    Trial_ts = []
    for Trials in range(0,Behav_data['SessionData']['nTrials']):#loop across trials (data files) 
        TrialStart_ts = Behav_data['SessionData']['TrialStartTimestamp'][Trials] 
        Trial_ts.append(TrialStart_ts)
    return(Trial_ts)

def extract_reward_times(Behav_data):
    #pulls out all reward times across session for each port, returns long list of timestamps  
    All_Reward_Times = []
    for Trials in range(0,Behav_data['SessionData']['nTrials']):#loop across trials (data files) 
        if ('Reward') in Behav_data['SessionData']['RawEvents']['Trial'][Trials]['States']:
            TrialStart_ts = Behav_data['SessionData']['TrialStartTimestamp'][Trials] 
            Rewardtime_offset = Behav_data['SessionData']['RawEvents']['Trial'][Trials]['States']['Reward'][0]
            R_times = (TrialStart_ts + Rewardtime_offset)
            R_times = [R_times]
            All_Reward_Times = All_Reward_Times + R_times

    return All_Reward_Times

def find_reward_inds(All_PortIn_Times_sorted,All_Port_references_sorted,Reward_ts):
    Rewarded_events = []
    counter = 0
    for index,ports in enumerate(All_Port_references_sorted):
        if ports == 7:
            if counter < len(Reward_ts):
                if np.isnan(Reward_ts[counter]):
                    counter = counter + 1
                if counter < len(Reward_ts):
                    if  All_PortIn_Times_sorted[index] >= Reward_ts[counter]:
                        Rewarded_events = Rewarded_events + [index]
                        counter = counter + 1
    return(Rewarded_events)

def extract_trial_end_times(Behav_data):
    #pulls out all end times across session for each trial, returns long list of timestamps  
    All_End_Times = []
    for Trials in range(0,Behav_data['SessionData']['nTrials']):#loop across trials (data files) 
        if ('ExitSeq') in Behav_data['SessionData']['RawEvents']['Trial'][Trials]['States']:
            TrialStart_ts = Behav_data['SessionData']['TrialStartTimestamp'][Trials] 
            Exittime_offset = Behav_data['SessionData']['RawEvents']['Trial'][Trials]['States']['ExitSeq'][-1]
            E_times = (TrialStart_ts + Exittime_offset)
            E_times = [E_times]
            All_End_Times = All_End_Times + E_times
        else:
            All_End_Times = All_End_Times + ['NaN']

    return All_End_Times

def determine_trial_id(All_PortIn_Times_sorted,Trial_end_ts):
    trial_id = []
    trial_number = 1
    for index,item in enumerate(All_PortIn_Times_sorted):
        if float(item) <= Trial_end_ts[trial_number - 1]:
            trial_id = trial_id + [trial_number]
        else:
            trial_number = trial_number + 1
            trial_id = trial_id + [trial_number]
    return trial_id

def align_trigger_to_index(trigger,trigger_index,all_timestamps):
#     trigger = np.asarray(trigger)
#     trigger = trigger[np.logical_not(np.isnan(trigger))]
#     trigger = list(trigger)
#     Output_array = ['NaN'] * len(all_timestamps)
#     for index, item in enumerate(trigger_index):
#         Output_array[item] = trigger[index]
    return Output_array

def align_trigger_to_index(trigger,trigger_index,all_timestamps):
    Output_array = ['NaN'] * len(all_timestamps)
    for index, item in enumerate(trigger_index):
        Output_array[item] = trigger[index]
    return Output_array

def find_trialstart_index(trial_id):
    trialstart_index = [0]
    for index,item in enumerate(trial_id):
        if index>0:
            if not item == trial_id[index-1]:
                trialstart_index = trialstart_index + [index]
    return trialstart_index

def align_trial_start_end_timestamps(trial_id,trialstart_index,Trial_start_ts):
    trial_ts_aligned = []
    counter = 0
    for i in range(len(trial_id)):
        if counter+1 < len(trialstart_index):
            if i == trialstart_index[counter+1]:
                counter = counter + 1
        if counter < len(Trial_start_ts):
            trial_ts_aligned = trial_ts_aligned + [Trial_start_ts[counter]]
    return trial_ts_aligned

def Find_TrialStart_and_Poke1_camera_inds(Camera_trig_states):

    where_ttl_changes = list(np.where(np.roll(Camera_trig_states,1)!=Camera_trig_states)[0])
    if where_ttl_changes[0] == 0:
        where_ttl_changes = where_ttl_changes[1::]
    Poke1_camera_inds = where_ttl_changes[1::2]
    Trial_start_camera_inds= where_ttl_changes[0::2]

    return Trial_start_camera_inds, Poke1_camera_inds

def generate_aligned_trial_end_camera_ts(Trial_start_camera_inds,trial_id,trialstart_index,Camera_ts):
    end_inds = []
    for index,item in enumerate(Trial_start_camera_inds):
        if index > 0:
            end_inds = end_inds + [item]

    Trial_end_Camera_Ts_aligned = align_trial_start_end_timestamps(trial_id,trialstart_index,Camera_ts[end_inds])
    last_trial_length = (len(trial_id) - trialstart_index[-1])
    if len(Trial_end_Camera_Ts_aligned) == len(trial_id): # this was a bit of a hacky fix, not rally sure about this - it all needs rewriting
        del Trial_end_Camera_Ts_aligned[(last_trial_length * -1)::] 
    Trial_end_Camera_Ts_aligned = Trial_end_Camera_Ts_aligned + (['NaN'] * last_trial_length )
    return Trial_end_Camera_Ts_aligned 

def align_firstpoke_camera_timestamps(trial_id,trialstart_index,Trial_start_ts,All_Port_references_sorted):
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


####### Part 2: ################



def Determine_Transition_Times_and_Types(All_PortIn_Times_sorted , All_Port_references_sorted):
    #transition times and types
    port_transition_types_temp = []
    port_transition_times_temp = []
    transition_reference_time = []
    for poke in range(1,len(All_PortIn_Times_sorted)):
        if (All_PortIn_Times_sorted[poke] - All_PortIn_Times_sorted[poke-1]) >= 0: 
            port_transition_times_temp = np.append(port_transition_times_temp,All_PortIn_Times_sorted[poke] - All_PortIn_Times_sorted[poke-1])
            port_transition_types_temp = np.append(port_transition_types_temp,Determine_transition_matrix(All_Port_references_sorted[poke-1],All_Port_references_sorted[poke]))
            transition_reference_time = transition_reference_time + [All_PortIn_Times_sorted[poke]]
    port_transition_types_temp = port_transition_types_temp.astype(int)
    return port_transition_times_temp, port_transition_types_temp,transition_reference_time        
             
        
        
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

def port_events_in_camera_time(trial_start_ts_aligned,Start_Port_time,Trial_start_Camera_Ts_aligned):

    port_camera_ts = []
    for index,item in enumerate(trial_start_ts_aligned[0:-1]):
        difference = Start_Port_time[index] - item
        port_camera_ts = port_camera_ts + [(Trial_start_Camera_Ts_aligned[index] + difference)]
        
    return port_camera_ts

############ Part 3:
def CreateSequences_Time(Transition_types,Transition_times,port1,transition_reference_time,Transition_filter_time):
    # reoder transitions into time releveant sequences  
    seq_index = 0
    TimeFiltered_ids = [[]]
    TimeFiltered_times = [[]]
    Reference_times = [[]]

    for ind, transit in enumerate (Transition_types):
        if Transition_times[ind] < Transition_filter_time and Transition_times[ind] > 0.03: # if less than filter time and more than lower bound filter time (0.05s):
            TimeFiltered_ids[seq_index] = TimeFiltered_ids[seq_index] + [transit]
            TimeFiltered_times[seq_index] = TimeFiltered_times[seq_index] + [Transition_times[ind]]
            Reference_times[seq_index] = Reference_times[seq_index] + [transition_reference_time[ind]]
            

        else:
            if TimeFiltered_ids[seq_index]: # if not empty 
                seq_index = seq_index + 1
                TimeFiltered_ids = TimeFiltered_ids + [[]]
                TimeFiltered_times = TimeFiltered_times +[[]] 
                Reference_times = Reference_times + [[]]
                TimeFiltered_ids[seq_index] = TimeFiltered_ids[seq_index] + [transit]
                TimeFiltered_times[seq_index] = TimeFiltered_times[seq_index] + [Transition_times[ind]]   
                Reference_times[seq_index] = Reference_times[seq_index] + [transition_reference_time[ind]]
    return TimeFiltered_ids,TimeFiltered_times,Reference_times
        
def CreateSequences_TimeandPort(Transition_types,Transition_times,port1,transition_reference_time,Transition_filter_time):
    #  reoder transitions into time and poke releveant sequences  
    seq_index = 0
    TimeFiltered_ids = [[]]
    TimeFiltered_times = [[]]
    Reference_times = [[]]

    for ind, transit in enumerate (Transition_types):
        if Transition_times[ind] < Transition_filter_time and Transition_times[ind] > 0.03: # if less than filter time and more than lower bound filter time (0.1s)
            if int(str(transit)[0]) == port1: # check if first port matches filter port
                if TimeFiltered_ids[seq_index]:
                    seq_index = seq_index + 1
                    TimeFiltered_ids = TimeFiltered_ids + [[]]
                    TimeFiltered_times = TimeFiltered_times +[[]]
                    Reference_times = Reference_times + [[]]
                TimeFiltered_ids[seq_index] = TimeFiltered_ids[seq_index] + [transit]
                TimeFiltered_times[seq_index] = TimeFiltered_times[seq_index] + [Transition_times[ind]]
                Reference_times[seq_index] = Reference_times[seq_index] + [transition_reference_time[ind]]

            else: 
                TimeFiltered_ids[seq_index] = TimeFiltered_ids[seq_index] + [transit]
                TimeFiltered_times[seq_index] = TimeFiltered_times[seq_index] + [Transition_times[ind]]
                Reference_times[seq_index] = Reference_times[seq_index] + [transition_reference_time[ind]]

        elif TimeFiltered_ids[seq_index]: # if not empty 
            seq_index = seq_index + 1
            TimeFiltered_ids = TimeFiltered_ids + [[]]
            TimeFiltered_times = TimeFiltered_times +[[]]
            Reference_times = Reference_times + [[]]

    return TimeFiltered_ids,TimeFiltered_times,Reference_times    


def number_of_rewarded_events(Reward_ts_aligned):
    counter = 0
    for item in Reward_ts_aligned:
        if not item == 'NaN':
            counter = counter + 1
    no_rewarded_events = counter
    return(no_rewarded_events)

def Sort_LED_intensites(port2,port3,port4,port5):
    LED_intensities = []
    for index,intensity in enumerate(port2):
        LED_intensities = LED_intensities + [[intensity] + [port3[index]] + [port4[index]] + [port5[index]]]
    return LED_intensities

def Sort_interRewards(port1,port2,port3,port4):
    IntermediateRewards = []
    for index,IR in enumerate(port1):
        IntermediateRewards = IntermediateRewards + [[IR] + [port2[index]] + [port3[index]] + [port4[index]]]
    return IntermediateRewards      
        
def FindandFixDroppedInOutTriggers(All_PortOut_Times,All_PortOut_references,All_Port_references,All_PortIn_Times):
    
    print('error check')
    All_Refs = []
    All_Out_Times = []
    All_In_Times = []

    for port in range(1,9):
        Out_refs = np.array(All_PortOut_references)
        Out_times = np.array(All_PortOut_Times)
        current_Outport_refs = np.where(Out_refs == port)
        Out = list(Out_times[current_Outport_refs])

        In_refs = np.array(All_Port_references)
        In_times = np.array(All_PortIn_Times)
        current_Inport_refs = np.where(In_refs == port)
        In = list(In_times[current_Inport_refs])

        if not (len(In) - len(Out)) == 0:
            if len(In) > len(Out):
                fixed = False
                for i in range(0,len(Out)):
                    if Out[i] >= In[i+1]:
#                         print('Dropped Triggers error1a fixed')
                        Out.insert(i,In[i] + 0.0001)
            if len(In) > len(Out):
                if not fixed == True:
#                     print('Dropped Triggers error1b fixed')
                    Out = Out + ['NaN']

            if len(In) < len(Out):
                fixed = False
                for i in range(0,len(In)):
#                     if In[i] <= Out[i+1]:
#                         print('Dropped Triggers error2a fixed')       
#                 if len(In) > len(Out):
#                     print('Dropped Triggers error2b fixed')
                    In = In + [(Out[-1] - 0.0001)]

        All_Refs = All_Refs + len(In) * [port] 
        All_Out_Times = All_Out_Times + Out
        All_In_Times = All_In_Times + In

    return All_In_Times, All_Out_Times, All_Refs

def FindTimestamps(filedate,CameraPath,CurrentAnimal):
#     Search for timestamps for given animal and session
    file_date = filedate[6:8] + filedate[4:6] + filedate[2:4]

    TimeStampsExist = False
    TimeStampPath = 'N/A'

    if os.path.isdir(CameraPath + CurrentAnimal):
        Dirs = sorted(os.listdir(CameraPath + CurrentAnimal))
        if file_date in Dirs:
            sub_dir = (os.listdir(CameraPath + CurrentAnimal+'\\'+ file_date))
            for file in sub_dir:
                if file[-2] == 's': #if its a .csv and not a avi
                    if file != '.DS_Store': #if file is not the weird hidden file 
                        if len(file) > 10: # crapy hacky way to determine if metadata saved in filename or not 
                            camerafiletime = file[-12:-4].replace("_", "")
                            if int(camerafiletime) < int(filedate[9:15]):   
                                TimeStampsExist = True
                                TimeStampPath = CameraPath + CurrentAnimal + '\\' + file_date + '\\' + file
                                break
                            else:
                                print('camera started after behvaiour so ts ignored')
                        else: #timestamp filename doesnt have metadata in it so we cant double check the starttime and have to assume it is fine : 
                            TimeStampsExist = True
                            TimeStampPath = CameraPath + CurrentAnimal + '\\' + file_date + '\\' + file
                        
    return TimeStampsExist, TimeStampPath
    
def align_opto_trials_to_dataframe(trial_id,executed_optotrials):
    counter = 0
    optotrials_aligned = []
    for index,item in enumerate(trial_id):
        if index > 0:
            if item == trial_id[index-1]:
                optotrials_aligned = optotrials_aligned + [executed_optotrials[counter]]
            else:
                counter = counter + 1
                optotrials_aligned = optotrials_aligned + [executed_optotrials[counter]]
        else:
            optotrials_aligned = optotrials_aligned + [executed_optotrials[counter]]
    return optotrials_aligned     


def time_sort(All_PortIn_Times,All_PortOut_Times,All_Port_references):
    sort_index = [np.argsort(np.array(All_PortIn_Times))]
    in_times_list = np.array(All_PortIn_Times, dtype=np.float)[sort_index]
    out_times_list = np.array(All_PortOut_Times, dtype=np.float)[sort_index]
    reference_list = np.array(All_Port_references)[sort_index]
    return in_times_list,out_times_list,reference_list

def extract_poke_times(Behav_data):
    
    #pulls out all port in times across session for each port, returns long list of timestamps  
    #Alignes them to trial start timestamps so that they port in times are across the whole session
    #Also returns port number references for each port in event in a list of references
    All_PortIn_Times = []
    All_PortOut_Times= []
    All_Port_references = []

    for port in range(1,9):
        PortInTimes = []
        PortOutTimes = []
        for Trials in range(0,Behav_data['SessionData']['nTrials']):#loop across trials (data files)  

            if ('Port' + str(port) + 'In') in Behav_data['SessionData']['RawEvents']['Trial'][Trials]['Events']:
                TrialStart_ts = Behav_data['SessionData']['TrialStartTimestamp'][Trials] 
                PortIn_ts_offset = Behav_data['SessionData']['RawEvents']['Trial'][Trials]['Events'][('Port' + str(port) + 'In')]
                PortIn_ts = (TrialStart_ts + PortIn_ts_offset)
                if type(PortIn_ts) is np.float64: 
                    PortIn_ts = [PortIn_ts]

                PortInTimes = PortInTimes + list(PortIn_ts)


            if ('Port' + str(port) + 'Out') in Behav_data['SessionData']['RawEvents']['Trial'][Trials]['Events']:
                TrialStart_ts = Behav_data['SessionData']['TrialStartTimestamp'][Trials] 
                PortOut_ts_offset = Behav_data['SessionData']['RawEvents']['Trial'][Trials]['Events'][('Port' + str(port) + 'Out')]
                PortOut_ts = (TrialStart_ts + PortOut_ts_offset)
                if type(PortOut_ts) is np.float64: 
                    PortOut_ts = [PortOut_ts]

                PortOutTimes = PortOutTimes + list(PortOut_ts)

        ## error chekc and fix (inset nan where event was dropped)
        if not len(PortInTimes) == len(PortOutTimes):
            PortInTimes,PortOutTimes = Error_check_and_fix(PortInTimes,PortOutTimes)

        All_Port_references = All_Port_references + (len(PortInTimes)*[port])
        All_PortIn_Times = All_PortIn_Times + PortInTimes
        All_PortOut_Times = All_PortOut_Times + PortOutTimes

    return All_PortIn_Times,All_PortOut_Times,All_Port_references

def Error_check_and_fix(In,Out):

    if not len(In) == len(Out):

        if len(In) > len(Out):
            fixed = False
            for i in range(0,len(Out)):
                if Out[i] >= In[i+1]:
                    Out.insert(i,'nan')
        if len(In) > len(Out):
            if not fixed == True:
                Out = Out + ['nan']

        if len(Out) > len(In):
            fixed = False
            for i in range(0,len(In)):
                if In[i] >= Out[i]:
                    In.insert(i,'nan')
        if len(Out) > len(In):
            if not fixed == True:
                Out = Out + ['nan']

    if not len(In) == len(Out):
        print('dropped event not fixed!!!!')
    
    return(In,Out)

def remove_dropped_in_events(All_PortIn_Times,All_PortOut_Times,All_Port_references):
    # we are trusting out events more than in events since outs are often dropped at teh end of trials. So only dropped 'in' events are removed. dropped out events are left as nan for now. 
    # these are sorted by in events and we cant sort nans so we have to remove these nans. 
    for index,item in enumerate(All_PortIn_Times):
        if item == 'nan': 
            del All_PortOut_Times[index]
            del All_PortIn_Times[index]
            del All_Port_references[index]
    return(All_PortOut_Times,All_PortIn_Times,All_Port_references)

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
