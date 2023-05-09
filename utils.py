import os 
import json
import numpy as np 


_obj = open(os.path.join(os.path.dirname(__file__),"data_folder", "all_widths.json"),'rt')
width_dict = json.load(_obj)
_obj.close()

def get_width(param, mode):
    subdict = width_dict[param]
    for entry in subdict:
        if str(mode).lower()==entry["mode0"]:
            return entry["width"]
    
    for entry in subdict:
        raise Exception()
        print(str(mode))
        
cor_file = os.path.join(os.path.dirname(__file__), "data_folder", "ice_covariance.json")
_obj = open(cor_file, 'rt')
cor_dict = json.load(_obj)
_obj.close()

param_types = ["Amp", "Phs"]
modes = [0,1,2,3,4]
all_widths = []
for param in param_types:
    for mode in modes:
        if param=="Phs" and mode==0:
            continue
        all_widths.append( get_width(param, mode) )
all_widths = np.array(all_widths)

# load correlation matrix in 
correlation = np.zeros(shape=(9,9))
for i, key in enumerate(cor_dict.keys()):
    for j, subkey in enumerate(cor_dict.keys()):
        correlation[i][j] = cor_dict[key][subkey] 