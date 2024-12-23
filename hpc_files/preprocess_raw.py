# %% [markdown]
# # import

# %%
import spikeinterface.full as si
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from probeinterface.plotting import plot_probe
import os

# %% [markdown]
# # set paths and choose probe 

# %%

path_ = r"/ceph/sjones/projects/sequence_squad/revision_data/lars_recordings/ephys//"

base_recording_paths = []
for q in os.listdir(path_):
    if not 'other_sessions' in q:
        folder = os.path.join(path_,q)
        for q in os.listdir(folder):
            base_recording_paths+=[os.path.join(folder,q)]
            
base_recording_paths

# %%
os.listdir(path_)

# %% [markdown]
# # make folder structure

# %%
all_recording_paths = []
for i in range(len(base_recording_paths)):
    
    organised_path = r"/ceph/sjones/projects/sequence_squad/revision_data/organised_data/animals//"

    mouse_id = base_recording_paths[i].split('/')[-1].split('_')[0]
    date_ = base_recording_paths[i].split('/')[-1].split('_')[1]
    #reverse the date
    date_ = '-'.join(date_.split('-')[::-1])

    if not os.path.isdir(organised_path+mouse_id+'_implant1'):
        print('adding mouse file')
        os.makedirs(organised_path+mouse_id+'_implant1')
    else:
        print('mouse file already exists')
        
    #work out what rnum will be
    make_folder = False
    if len(os.listdir(organised_path+mouse_id+'_implant1//')) == 0:
        r_num = '1'
        make_folder = True
    else:
        r_num = str(int(os.listdir(organised_path+mouse_id+'_implant1//')[-1].split('_')[0][-1]) + 1)
        for item in os.listdir(organised_path+mouse_id+'_implant1//'):
            if not date_ in item:
                make_folder = True
            else:
                make_folder = False
                r_num = item.split('_')[0][-1]
                break
        
    # make recording dir
    if make_folder == True:
        print('**making new recording folder')
        recording_path = organised_path+mouse_id+'_implant1' + '//recording' + r_num + '_' + date_ + '//'
        os.makedirs(recording_path)
        # make folder structure 
        if not os.path.isdir(recording_path + 'ephys'):
            os.makedirs(recording_path + 'ephys')
        if not os.path.isdir(recording_path + 'video/tracking/'):
            os.makedirs(recording_path + 'video/tracking/')
        if not os.path.isdir(recording_path + 'behav_sync/'):
            os.makedirs(recording_path + 'behav_sync/') 
            

    else:
        print('recording folder already exists')
        recording_path = organised_path+mouse_id+'_implant1' + '//recording' + r_num + '_' + date_ + '//'
    all_recording_paths+=[recording_path + 'ephys']
            


# %% [markdown]
# # chose which data to process - ie. data that hasnt yet been processed - and create save paths

# %%

process_base_paths  = []
out_paths = []
missing_probe = []
for index,path in enumerate(all_recording_paths):
    if not os.path.isdir(path+ '/probeA/kilosort4_output//'):
        # if no kilosort folder then add base path to process list
        process_base_paths += [base_recording_paths[index]]
        out_paths += [path]
        missing_probe += ['A']

    if not 'ap5' in path:
        if not os.path.isdir(path+ '/probeB/kilosort4_output//'):
            # if no kilosort folder then add base path to process list
            process_base_paths += [base_recording_paths[index]]
            out_paths += [path]
            missing_probe += ['B']

# %% [markdown]
# # kilosort loop

# %%
## reverse for this script, just so I can run two of these scritps at once without them conflictiing
process_base_paths = process_base_paths[::-1]
out_paths = out_paths[::-1]

for ind,base_folder in enumerate(process_base_paths):
    print(str(ind) + 'out of' + str(len(process_base_paths)))
    print('*********** PROCESSING: ***********')
    print(base_folder)
    out_path = out_paths[ind]

    # extract the stream names (each np processor)
    stream_names, stream_ids = si.get_neo_streams('openephysbinary', base_folder)
    print(stream_names)

    # chose probe id, DEAL WITH PROBE A/B stuff
    ksort = False
    for stream_i, stream in enumerate(stream_names):
        if 'Probe' + missing_probe[ind] in stream:
            if not 'LFP' in stream:
                Probe_id = stream_names[stream_i]
                out_path = out_path + '//probe' + missing_probe[ind] + '//'
                out_path_object = Path(out_path)
                ksort = True
                if not os.path.isdir(out_path):
                    os.makedirs(out_path)
                break

    if ksort == True:
        
        # load in data
        raw_rec = si.read_openephys(base_folder,stream_name = Probe_id,load_sync_channel=False)
        raw_rec.get_probe().to_dataframe()
    
        # plot probe
        fig, axs = plt.subplots(figsize=(1, 100))
        probe = raw_rec.get_probe()
        plot_probe(probe, ax = axs)
        plt.savefig(out_path + 'probe_map.png')
        plt.close()
            
        
        # Preprocess the recording¶
        # Let’s do something similar to the IBL destriping chain (See :ref:ibl_destripe) to preprocess the data but:
        # instead of interpolating bad channels, we remove then.
        # instead of highpass_spatial_filter() we use common_reference()
        
        rec1 = si.highpass_filter(raw_rec, freq_min=400.)
        bad_channel_ids, channel_labels = si.detect_bad_channels(rec1)
        rec2 = rec1.remove_channels(bad_channel_ids)
        print('bad_channel_ids', bad_channel_ids)
    
        rec3 = si.phase_shift(rec2)
        rec4 = si.common_reference(rec3, operator="median", reference="global")
        rec = rec4
        
        
        ## save out the preprocessed binary
    
        job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
        rec = rec.save(folder=out_path_object / 'preprocess', format='binary', **job_kwargs)
        
        # here we use static plot using matplotlib backend
        fig, axs = plt.subplots(ncols=3, figsize=(20, 10))
    
        si.plot_traces(rec1, backend='matplotlib',  clim=(-50, 50), ax=axs[0])
        si.plot_traces(rec4, backend='matplotlib',  clim=(-50, 50), ax=axs[1])
        si.plot_traces(rec, backend='matplotlib',  clim=(-50, 50), ax=axs[2])
        for i, label in enumerate(('filter', 'cmr', 'final')):
            axs[i].set_title(label)
        plt.savefig(out_path + 'preprocessing_destriping_common_ref.png')
        plt.close()
        
        
        # plot some channels
        fig, ax = plt.subplots(figsize=(20, 10))
        some_chans = rec.channel_ids[[100, 150, 200, ]]
        si.plot_traces({'filter':rec1, 'cmr': rec4}, backend='matplotlib', mode='line', ax=ax, channel_ids=some_chans)
        plt.savefig(out_path + 'example_chans.png')
        plt.close()
        
        # check noise 
        # we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
        noise_levels_microV = si.get_noise_levels(rec, return_scaled=True)
        noise_levels_int16 = si.get_noise_levels(rec, return_scaled=False)
        
        fig, ax = plt.subplots()
        _ = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
        ax.set_xlabel('noise  [microV]')
        plt.savefig(out_path + 'noise_level.png')
        plt.close()
        
        
        # check default params for kilosort2.5
        params_kilosort4 = si.get_default_sorter_params('kilosort4')
        params_kilosort4['delete_recording_dat'] = False
    
        # # run kilosort4 with drift correction (set as True in the params)
        sorting = si.run_sorter('kilosort4', rec, output_folder=out_path_object / 'kilosort4_output',
                                docker_image=False, verbose=True, **params_kilosort4)
        
        
        ######################################################################################
        # load back in to check quality
        sorting = si.read_sorter_folder(out_path_object / 'kilosort4_output')
        
        analyzer = si.create_sorting_analyzer(sorting, rec, sparse=True, format="memory")
        
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
        analyzer.compute("waveforms",  ms_before=1.5,ms_after=2., **job_kwargs)
        analyzer.compute("templates", operators=["average", "median", "std"])
        analyzer.compute("noise_levels")
    
        analyzer_saved = analyzer.save_as(folder=out_path_object / "analyzer", format="binary_folder")
    
        metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff']
    
        metrics = si.compute_quality_metrics(analyzer, metric_names=metric_names)
    
        amplitude_cutoff_thresh = 0.1
        isi_violations_ratio_thresh = 1
        presence_ratio_thresh = 0.9
    
        our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
    
        keep_units = metrics.query(our_query)
        keep_unit_ids = keep_units.index.values
    
        analyzer_clean = analyzer.select_units(keep_unit_ids, folder=out_path_object / 'analyzer_clean', format='binary_folder')
    
        # export spike sorting report to a folder
        si.export_report(analyzer_clean, out_path_object / 'report', format='png')

        ### SAVE OUT A TXT FILE WITH NUMBER OF UNITS DATA ON IT
        # Open a file in write mode
        file_path = out_path + 'unit_info.txt'
        with open(file_path, "w") as file:
            # Use the file argument to save print output to the file
            print('Kilosort output:',file = file)
            print(sorting, file=file)
            file.write("/n")
            print('"good" units:',file = file)
            print(analyzer_clean, file=file)

        print('DONE!')
    else:
        print('Already processed or no probe B!')

print('ALL DONE!')


