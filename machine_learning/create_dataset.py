
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import csv



def main():

    raw_solves_data = grab_json_file('data/filteredSolves.json')

    raw_studies_data = grab_json_file('data/filteredStudies.json')

    solve_data, solve_data_key_name = new_data(raw_solves_data, 'iteration_number')

    studies_data, studies_data_key_name = new_data(raw_studies_data, '')

    solvetime_data = grab_solvetimes_file('data/solveTimes.txt')

    ##############################################
    ### voxels seems promising
    # entry = ['voxels']
    # entry_name = 'test'
    entry = ['number_of_keep_outs', 'number_of_loads', 'number_of_geometries', 'number_of_keep_ins']
    entry_name = 'classify_four_v2'

    dataset = generate_dataset(studies_data, solvetime_data, name = 'study', entry = entry, file_path = 'data')
    # dataset = generate_dataset(solve_data, solvetime_data, name = 'solve', entry = entry, file_path = 'data')

    dataset = classify_solvetimes(dataset)

    # dataset = np.asarray(dataset)

    # dataset = normalize(dataset)

    print(dataset)
    # print(dataset[:, 0])
    # print(dataset[:, 1])

    # plt.scatter(dataset[:, 0], dataset[:, 1])
    # plt.show()
    #############################################

    # write_dataset_to_file(dataset, entry_name)

    write_dataset_to_file(dataset, entry_name)


    # To check if pickle worked properly
    # file_name = entry_name + '_dataset.p'
    # with open ('data/' + file_name, 'rb') as fp:
    #     itemlist = pickle.load(fp) 
    
    # print(itemlist)
    # print(itemlist.shape)

def classify_solvetimes(dataset):
    count_of_short = 0
    count_of_med   = 0
    count_of_long  = 0

    for i in range(len(dataset)):
        if i == 0:
            continue

        if dataset[i][-1] <= 2:
            count_of_short += 1
            dataset[i][-1] = 0
            # dataset[i][-1] = 0
            # dataset[i][-2] = 0
            # dataset[i][-3] = 1
        elif dataset[i][-1] > 2 and dataset[i][-1] <= 7:
            count_of_med += 1
            dataset[i][-1] = 1
            # dataset[i][-1] = 0
            # dataset[i][-2] = 1
            # dataset[i][-3] = 0
        elif dataset[i][-1] > 7:
            count_of_long += 1
            dataset[i][-1] = 2
            # dataset[i][-1] = 1
            # dataset[i][-2] = 0
            # dataset[i][-3] = 0
    
    print("count of short: {}".format(count_of_short))
    print("count of med: {}".format(count_of_med))
    print("count of long: {}".format(count_of_long))
    return dataset


def normalize(dataset):
    # print(dataset)
    max_list = np.amax(dataset, axis=0)
    y_entry = len(max_list) - 1
    max_list[y_entry] = 1.
    # print(max_list)
    new_dataset = dataset[:-1] / max_list
    new_dataset = np.around(new_dataset, 3)
    return new_dataset

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
                if entry[key_name] == 0:
                    data[name].append(entry)
 
    return data, name


def generate_dataset(x_data, y_data, name, entry, file_path):

    entry_names = entry
    entry_names.append('time')
    print(entry_names)
    dataset = []
    dataset.append(entry_names)

    if name == 'solve':
        match_id = 'solver_study_id'
    else:
        match_id = 'id'

    for x_entry in x_data[name]:
        group = []
        for y_entry in y_data:
            if x_entry[match_id] == y_entry[0]:
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

    # file_name = entry_name + '_dataset.p'
    # with open('data/' + file_name, 'wb') as fp:
    #     pickle.dump(dataset, fp)
    file_name = entry_name + '_dataset.csv'
    with open('data/' + file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dataset)


def grab_json_file(file_path):

    with open(file_path, 'r', encoding='utf8', errors='ignore') as json_file:
        data = json.load(json_file)
    
    return data
 
def grab_solvetimes_file(file_path):

    solve_file = open(file_path, "r")

    shit_way_to_trim_short_entries = 0

    solve_data = []
    for line in solve_file:
        entry = line.split()
        entry[1] = round((float(entry[1]) / 3600), 3)
        if entry[1] >= 0.5:
            # if shit_way_to_trim_short_entries == 0:
            solve_data.append(entry)
                # shit_way_to_trim_short_entries = 1
            # else:
                # shit_way_to_trim_short_entries = 0

    return solve_data


if __name__ == "__main__":
    main()