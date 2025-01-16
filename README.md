Emmett James Thompson
# Neuropixel data preprocessing
scripts for processing data collected during ephys recordings for the 8 port sequence task

- See our publication for more details: [Replay of Procedural Experience is Independent of the Hippocampus](https://www.biorxiv.org/content/10.1101/2024.06.05.597547v1.full.pdf).

# Overview: 
- there are three main data streams (ephys, behaviour and video data)
- each of these need to be processed intially (scripts 1-5)
- then finally these data streams are aligned and a synchronisation file is created=
- these scripts produce processed data in a clean organised file structure
  
![Processing pipeline](images/processing_schematic.png)


# Data collection:
This processing pipeline assumes the following data collection methods:
- Neuropixels (1.0 or 2.0) collected using Open Ephys
- Behaviour data collected with Bpod using sequence task protocol "Sequence Automated"
- FLIR Cameras aligned above and below the task/sleep arenas, set with GPIO receiveing triggers, avi and stamps data saved via bonsai
- Synchronisation (see TTL alignment below)

# Processing Pipeline

### the code is organized into 6 phases (based on 6 notebook scripts) that should be executed in order:

---

## 1. Ephys processing
- Processes the raw electrophysiology (ephys) data (Open Ephys output) using the **SpikeInterface** architecture.
- Creates a probe object for each probe (active channel map) and spikesorts the data using **Kilosort 4**.

#### Requirements:
- [SpikeInterface](https://spikeinterface.readthedocs.io/en/stable/)
- [Kilosort 4](https://github.com/MouseLand/Kilosort)
  - *Note*: Kilosort is computationally intensive but performs significantly faster with GPU access.
  - Refer to the `HPC_helpsheet` file for tips on running this step on a computing cluster.

#### Output:
- Organized file directories for each recording.
- kilosort output (spike times) for each probe

---

## 2. Video processing
- Processes the video output from Bonsai (.avi video files and .csv timestamp files):
  1. Converts timestamps into seconds.
  2. Separates and labels the three experimental phases of the videos using trigger times (see **TTL Alignment**).
  3. automaticaly detects and labels video type (back view or above view camera) 

#### Output:
- renames and copies the raw `.avi` files into the organized file directory.
- Copies this same file into a dump folder (used in the next stage as the source directory for running tracking with DeepLabCut).
- creates a camera timstamp dataframe file for each video 

---

## 3. Video tracking
- Create tracking files for each video
- 1. notebook script creates `.sh` shell script files which can be executed on the cluster to perform deeplabcut tracking 
- 2. use DLC to create 
- 
#### Requirements:
- [Deeplabcut](https://deeplabcut.github.io/DeepLabCut/README.html)
- Refer to the `HPC_helpsheet` file for tips on running this step on a computing cluster.

#### output
- Tracking files (behavioural port and mouse head centre) for each video in the organised directory

---

## 4. Bpod processing
- Notebook script takes the raw `.mat` ouput from bpod, processes this data and converts the values into a useful python readable format

#### output
- processed behavioural data files which are stored in the organised data directory

---

## 5. Spectral alignment / histology (Probe location)
-  The aim of this step is the know where in the brain each kilsorted unit was located.
-  This can be done based on spectral analysis of the ephys data (what the signal looked like) or by looking at the probe trace in the brain (or both together)

- PROBE SPECTRAL ANALYSIS: running the script provided should perform this analysis
  ** note: you may need multiple recordings across the entire length of the probe for this analysis to be useful.

- HISTOLOGY


## 6. Data Alignment 

---


  

  

# TTL alignment (synchronisartion) structure: 

Bpod (behvaioural controller) acts as the central clock, sending TTLs to the ephys  niDAQ and the camera GPIO pins. 
![Processing pipeline](images/ttl_clock.png)
The recording epoch is split into sleep and task periods. During sleep Bpod acts like a heart beat, sending pulses every 30s. During the task, TTLs are sent based on the trial structure (ITI depends on the behaviur of the mouse)
![Processing pipeline](images/TTL_task_structure.png)
TTL output during the task goes high at trial start (when bpod loads a new trial), and stays high until the mouse pokes into the first task port in the sequence (ie. port 1 of 5, not just any port!) 
![Processing pipeline](images/task_ttl_relationship.png)

## Getting Started

Ensure you have the following software installed:
- [Git](https://git-scm.com/)
- [Python](https://www.python.org/downloads/)  (Version used: 3.12.3)
- Jupyter notebook
- Necessary Python libraries: this pipeline worked with the environment I have listed in requirements.txt (though some of these packages may be redundant..also some may be missing, but they should be easy to find!)
- IMPORTANT! make sure to use moviepy version 1.0.3. the newer one at time of writing (december 2024) seems to be broken 

