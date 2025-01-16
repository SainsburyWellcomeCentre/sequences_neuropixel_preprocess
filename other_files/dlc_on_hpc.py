import deeplabcut as dlc
import sys
from pathlib import Path
import shutil
import os 

for arg in sys.argv[1:-2]:
    if not Path(arg).exists():
        print(arg)
        print('NOT FOUND')
        raise FileNotFoundError(arg)
    
# if not is dir, make dir:
if not Path(sys.argv[3]).exists():
    Path(sys.argv[3]).mkdir(parents=True, exist_ok=True)

print('starting tracking for real now')
dlc.analyze_videos(sys.argv[1], [str(sys.argv[2])], destfolder=sys.argv[3])
print('Done, hopefully...')


## make videos
if sys.argv[4] == 'TRUE':
    print('************ making videos ****************')

    config_path = str(sys.argv[1])
    destfolder =  sys.argv[3] + r'/test_videos//'
    videos = destfolder + '/' + Path(sys.argv[2]).parts[-1]
        
    # if not is dir, make dir:
    if not Path(destfolder).exists():
        Path(destfolder).mkdir(parents=True, exist_ok=True)
        
    ## copy videos and tracking to test folder:
    print('copying video' + sys.argv[2])
    shutil.copy(sys.argv[2], destfolder)
    
    for file in os.listdir(sys.argv[3]):
        # if the file is a file and not a directory:
        copy_file = os.path.join(sys.argv[3], file)
        if os.path.isfile(copy_file):
            print('copying' + copy_file)
            shutil.copy(copy_file, destfolder)

    print('----------------------------------------------------------------------------------')
    print('config_path:', config_path)
    print('videos:', videos)
    print('destfolder:', destfolder)
    print('----------------------------------------------------------------------------------')

    dlc.create_labeled_video(config_path, videos, save_frames = False)


