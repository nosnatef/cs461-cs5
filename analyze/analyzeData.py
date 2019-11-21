
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


def main():

    all_data = grab_json_file('data/filteredSolves.json')

    studies = grab_json_file('data/filteredStudies.json')

    data, data_key_name = new_data(all_data, 'final_iteration')

    studies_data, studies_data_key_name = new_data(studies)

    longest_key = grab_longest_key(data, data_key_name)

    longest_studies_key = grab_longest_key(studies_data, studies_data_key_name)

    dict_count = count_key_entries(data, data_key_name, longest_key)

    studies_dict_count = count_key_entries(studies_data, studies_data_key_name, longest_studies_key)

    for k, v in dict_count.items():
        print("{:<30} = {}".format(k, v))

    print("*****************************************")

    for k, v in studies_dict_count.items():
        print("{:<30} = {}".format(k, v))

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

    longest = 0
    longest_key = []
    for one_key in all_keys:
        if len(one_key) == longest:
            print("*NOTE*: Missing key with same length as longest key in dictionary!")
        elif len(one_key) > longest:
            longest_key = one_key

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



def grab_json_file(file_path):

    with open(file_path, 'r', encoding='utf8', errors='ignore') as json_file:
        data = json.load(json_file)
    
    return data
   
if __name__ == "__main__":
    main()
