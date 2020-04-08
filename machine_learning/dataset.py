
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle



def main():

    raw_solves_data = grab_json_file('data/filteredSolves.json')

    raw_studies_data = grab_json_file('data/filteredStudies.json')

    finaliter_solve_data, finaliter_solve_data_key_name = new_data(raw_solves_data, 'final_iteration')

    # print(finaliter_solve_data)

    studies_data, studies_data_key_name = new_data(raw_studies_data, '')

    # for entry in studies_data['study']:
    #     print("*******************************")
    #     for k, v in entry.items():
    #         print("{:<30} = {}".format(k, v))

    # the entry 'id' is where the solveTimes refer to

    studies_solve_data = grab_solvetimes_file('data/solveTimes.txt')

    # entry = ['number_of_keep_outs', 'number_of_loads', 'number_of_geometries', 'number_of_keep_ins']
    ##############################################
    # voxels seems promising
    entry = ['target_volume']
    entry_name = 'test'

    # dataset = generate_dataset(studies_data, studies_solve_data, entry = entry, file_path = 'data')
    dataset = generate_dataset(finaliter_solve_data, studies_solve_data, name = 'study', entry = entry, file_path = 'data')

    dataset = np.asarray(dataset)

    print(dataset)
    print(dataset[:, 0])
    print(dataset[:, 1])

    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.show()
    #############################################

    # write_dataset_to_file(dataset, entry_name)


    # To check if pickle worked properly
    # file_name = entry_name + '_dataset.p'
    # with open ('data/' + file_name, 'rb') as fp:
    #     itemlist = pickle.load(fp) 
    
    # print(itemlist)
    # print(itemlist.shape)


def new_data(old_data, key_name):
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


def generate_dataset(x_data, y_data, name, entry, file_path):

    dataset = []

    for x_entry in x_data['solve']:
        group = []
        for y_entry in y_data:
            if x_entry['solver_study_id'] == y_entry[0]:
                for entry_name in entry:
                    if entry_name in x_entry:
                        if entry_name == "voxels":
                            group.append(math.log(x_entry[entry_name]))
                        else:
                            group.append(x_entry[entry_name])
                
                group.append(y_entry[1])
        if len(group) > 1:
            dataset.append(group)

    print(dataset)
    return dataset

def write_dataset_to_file(dataset, entry_name):

    # with open('data/dataset.txt', 'w+') as f:
    #     for entry in dataset:
    #         f.write(entry)

    file_name = entry_name + '_dataset.p'
    with open('data/' + file_name, 'wb') as fp:
        pickle.dump(dataset, fp)


def grab_json_file(file_path):

    with open(file_path, 'r', encoding='utf8', errors='ignore') as json_file:
        data = json.load(json_file)
    
    return data
 
def grab_solvetimes_file(file_path):

    solve_file = open(file_path, "r")

    solve_data = []
    for line in solve_file:
        entry = line.split()
        entry[1] = round(float(entry[1]) / 3600)
        # print(entry[1])
        solve_data.append(entry)

    return solve_data


if __name__ == "__main__":
    main()