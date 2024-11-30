import numpy as np
def concatinate(Data):
    for sessions, trials in enumerate(Data):
        print(sessions)
        if sessions == 0:
            Concat_Data = trials
            print(Concat_Data)
        else:
           ConcData = [Concat_Data[-1]] + trials
           Concat_Data = np.append(Concat_Data,ConcData)
    return Concat_Data