
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import csv



# This function loads in the data and then goes through it to grab the entries that are relevenat to train on for the model
# basically compiling the dataset that the machine learning model will train on
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
    study_entries = ['number_of_keep_outs', 'number_of_loads', 'number_of_load_cases', 'number_of_geometries', 'number_of_keep_ins']
    solve_entries = ['voxels']
    entry = {'study': study_entries,
             'solve': solve_entries}

    entry_name = 'combined_v3'
    print(entry)

    dataset = generate_dataset(studies_data, solve_data, solvetime_data, entry = entry, file_path = 'data')

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

# classify_solvetimes - goes through the time entries and transforms them into the label assiocated with that time frame
#                       short = 0-2hours, medium = 2-7hours, long = 7+ hours
#   dataset - the dataset that is being manipulated and transformed
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


# normalized - function that was normalizing the inputs but not really using anymore. Basically divide all inputs by avg.
def normalize(dataset):
    # print(dataset)
    max_list = np.amax(dataset, axis=0)
    y_entry = len(max_list) - 1
    max_list[y_entry] = 1.
    # print(max_list)
    new_dataset = dataset[:-1] / max_list
    new_dataset = np.around(new_dataset, 3)
    return new_dataset

# new_data - just takes in a json dataset and converts it into a python dictionary
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

# generate_dataset - takes in the new datasets and compiles everything into the true dataset that the model will learn on
#   x_studies_data - the dictionary dataset from the studies dataset
#   x_solved_data - the dictionary dataset form the solves dataset
#   y_data - the time entries from the solveTimes.txt dataset
#   entry - the entry_names that we want to extract from each dataset
#   file_path - where the file will be saved and called
def generate_dataset(x_studies_data, x_solves_data, y_data, entry, file_path):
    entry_names = []
    entry_names.extend(entry['study'])
    entry_names.extend(entry['solve'])
    entry_names.append('time')
    print(entry_names)
    study_dataset = []

    for x_study_entry in x_studies_data['study']:
        id_study_obj = x_study_entry['id']

        group = []
        for y_entry in y_data:
            if id_study_obj == y_entry[0]:
                group.append(id_study_obj)
                for study_entry_name in entry['study']:
                    group.append(x_study_entry[study_entry_name])
                group.append(y_entry[1])
                break

        if len(group) == len(entry['study']) + 2:
            study_dataset.append(group)
        # else:
        #     print("WOWOW")

    solve_dataset = []
    # solve_dataset.append(entry_names)

    for x_solve_entry in x_solves_data['solve']:
        if x_solve_entry['solver_study_status'] == "DONE" and 'voxels' in  x_solve_entry and x_solve_entry['manufacturing_method'] != "Frame":
            id_solve_obj = x_solve_entry['solver_study_id']

            group = []
            for y_entry in y_data:
                if id_solve_obj == y_entry[0]:
                    group.append(id_solve_obj)
                    for solve_entry_name in entry['solve']:
                        group.append(round(math.log(x_solve_entry[solve_entry_name]), 5))
            if len(group) == len(entry['solve']) + 1:
                solve_dataset.append(group)

    true_dataset = []
    true_dataset.append(entry_names)

    for study_entry in study_dataset:
        group = []
        for solve_entry in solve_dataset:
            if study_entry[0] == solve_entry[0]:
                group.extend(study_entry[1:-1])
                group.append(solve_entry[-1])
                group.append(study_entry[-1])
                true_dataset.append(group)

    # for i in true_dataset[1:]:
    #     obj_id = i[0]
    #     count = 0
    #     for j in true_dataset:
    #         other_obj_id = j[0]
    #         if obj_id == other_obj_id:
    #             count += 1
    #             if count > 1:
    #                 print("FUDGYWUDGY")
    
    # print(true_dataset)




#########################################################
                    # for study_entry in study_dataset:
                    #     if id_solve_obj == study_entry[0]:
                    #         for solve_entry_name in entry['solve']:
                    #             # if solve_entry_name == "voxels":
                    #                 # print(x_solve_entry)
                    #             group.append(round(math.log(x_solve_entry[solve_entry_name]), 5))
                    #         # else:
                    #     #     group.append(x_solve_entry[solve_entry_name])

                    # if len(group) == len(entry['solve']):
                    #     truth = []
                    #     truth.extend(study_entry[:-1])
                    #     truth.extend(group)
                    #     truth.append(study_entry[-1])
                    #     true_dataset.append(truth)
########################################################
                    # print(x_solve_entry)
            # print(true_dataset)
        # else:
        #     print("wowowow")
        # count += 1
        # if count == 50:
        #     break

    print("################")
    # print(true_dataset)
    # print(study_dataset)
    # print(solve_dataset)

    return true_dataset

# def generate_dataset(x_data, y_data, name, entry, file_path):

    # entry_names = entry
    # entry_names.append('time')
    # print(entry_names)
    # dataset = []
    # dataset.append(entry_names)

    # if name == 'solve':
    #     match_id = 'solver_study_id'
    # else:
    #     match_id = 'id'

    # for x_entry in x_data[name]:
    #     group = []
    #     for y_entry in y_data:
    #         if x_entry[match_id] == y_entry[0]:
    #             for entry_name in entry:
    #                 if entry_name in x_entry:
    #                     if entry_name == "voxels":
    #                         group.append(math.log(x_entry[entry_name]))
    #                     else:
    #                         group.append(x_entry[entry_name])
                
    #             group.append(y_entry[1])
    #     if len(group) > 1:
    #         dataset.append(group)

    # print(dataset)
    # return dataset

# write_dataset_to_file - basically saves off the dataset being passed in
#   dataset - what data is being saved
#   entry_names - what the file will be called
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


# grab_json_file - from the file_path just load the json data and return it
def grab_json_file(file_path):

    with open(file_path, 'r', encoding='utf8', errors='ignore') as json_file:
        data = json.load(json_file)
    
    return data
 
 # grab_solvetimes_file - as states from file_path grabs and loads in the solve_times data
 #                        is slightly trimmed and modified to get more realistic values
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