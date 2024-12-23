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

# Code:
The code is organised into 6 scripts which should be executed in order:
- 1: this script...
- 2: this script...

# TTL alignment (synchronisartion) structure: 

Bpod (behvaioural controller) acts as the central clock, sending TTLs to the ephys niDAQ and the camera GPIO pins. 
![Processing pipeline](images/ttl_clock.png)
The recording epoch is split into sleep and task periods. During sleep Bpod acts like a heart beat, sending pulses every 30s. During the task, TTLs are sent based on the trial structure (ITI depends on the behaviur of the mouse)
![Processing pipeline](images/TTL_task_structure.png)
TTL output during the task goes high at trial start (when bpod loads a new trial), and stays high until the mouse pokes into the first task port in the sequence (ie. port 1 of 5, not just any port!) 
![Processing pipeline](images/task_ttl_relationship.png)
