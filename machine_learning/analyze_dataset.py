
# key_list = ['aggregate_object_name', 'iteration_number', 'minimum_member_thickness', 'manufacturing_method', 'solver_type',
#             'final_iteration', 'solver_study_id', 'maximum_von_mises_stress', 'derivatives',
#             'minimum_factor_of_safety', 'mass', 'volume', 'maximum_overhang_angle', 'solver_study_status',
#             'maximum_displacement', 'bucket_environment', 'object_name'
# ]
# missing_keys =  ['surface_VMS', 'internal_iteration_number', 'target_volume', 'convergence', 'voxels', 'size']

# ['aggregate_object_name', 'iteration_number', 'object_name', 'surface_VMS', 'maximum_von_mises_stress', 
# 'mass', 'maximum_overhang_angle', 'internal_iteration_number', 'target_volume', 'minimum_factor_of_safety', 
# 'bucket_environment', 'convergence', 'minimum_member_thickness', 'manufacturing_method', 'solver_type', 
# 'voxels', 'final_iteration', 'related_geometries', 'solver_study_id', 'volume', 'size', 'solver_study_status', 
# 'maximum_displacement']


# | # of load, # of load cases, # of constraints, # of geometries, # of keep in/out

import json
import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():

    all_data = grab_json_file('data/filteredSolves.json')

    studies = grab_json_file('data/filteredStudies.json')

    data, data_key_name = new_data(all_data, 'final_iteration')

    studies_data, studies_data_key_name = new_data(studies)

    longest_key = grab_longest_key(data, data_key_name)

    longest_studies_key = grab_longest_key(studies_data, studies_data_key_name)

    dict_count = count_key_entries(data, data_key_name, longest_key)

    studies_dict_count = count_key_entries(studies_data, studies_data_key_name, longest_studies_key)

    output_key_entries(dict_count, studies_dict_count)
   
    see_graph(studies_data)

    unit_test_dataset()
    
def unit_test_dataset():
    # name = 'number_of_loads_dataset'
    # name = 'number_of_geometries_dataset'
    # name = 'four_dataset'
    name = 'four_normalized_dataset'
    # name = 'test_dataset'
    entry = name + '.p'

    with open ('data/' + entry, 'rb') as fp:
        dataset = pickle.load(fp) 
 
    print(dataset.shape)
    print(dataset)

    for entry in dataset:
        for element in entry:
            assert type(element) != type(int), "Error: invalid entry in dataset. Should be all numbers"
            assert element >= 0, "Error: All entries should be a positive number"
            assert element < 100000, "Error: entry in dataset too large"




def count_key_entries(data, data_key_name, longest_key):
    dict_count = {}
    for key in longest_key:
        dict_count[key] = 0

    for entry in data[data_key_name]:
        for key in longest_key:
            if key in entry:
                dict_count[key] += 1

    return dict_count
 

def grab_longest_key(data, data_key_name):
    all_keys = []
    for entry in data[data_key_name]:
        if entry.keys() not in all_keys:
            all_keys.append(entry.keys())

    longest_key = []
    for one_key in all_keys:
        if len(one_key) > len(longest_key):
            longest_key = one_key

    for key in all_keys:
        if len(key) != len(longest_key):
            "*Note*: Inconsistent keys in file"

    # print(len(longest_key))
    # print(longest_key)
    return longest_key

def new_data(old_data, key_name=''):
    data = {}
    name = ''
    count = 0

    if key_name == '':
        name = 'study'
        data[name] = []
        for entry in old_data:
            data[name].append(entry)
 
    else:
        name = 'solve'
        data[name] = []
        for entry in old_data:
            if key_name in entry:
                data[name].append(entry)
 
    return data, name

def see_graph(data):
    # analysis of keep_in, keep_outs, # of geometries, # of loads
    hist_count = []
    hist_name = 'number_of_keep_ins'

    for entry in data['study']:
        if hist_name in entry:
            hist_count.append(entry[hist_name])
        else:
            hist_count.append(0)
    hist_count = np.asarray(hist_count)
    # print(hist_count[:5])

    hist, bin_edges = np.histogram(hist_count)
    # print(hist)
    # print(bin_edges)

    n, bins, patches = plt.hist(x=hist_count, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
                            # rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    # plt.grid(axis='y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(hist_name)
    # plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def output_key_entries(dict_count, studies_dict_count):
    for k, v in dict_count.items():
        print("{:<30} = {}".format(k, v))

    print("*****************************************")

    for k, v in studies_dict_count.items():
        print("{:<30} = {}".format(k, v))

    print("*****************************************\n\n")
 

def grab_json_file(file_path):

    with open(file_path, 'r', encoding='utf8', errors='ignore') as json_file:
        data = json.load(json_file)
    
    return data
   
if __name__ == "__main__":
    main()
