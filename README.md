# neuropixel data preprocessing
scripts for processing data collected during ephys recordings for the 8 port sequence task

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


![Processing pipeline](images/ttl_clock.png)
![Processing pipeline](images/TTL_task_structure.png)
![Processing pipeline](images/task_ttl_relationship.png)
