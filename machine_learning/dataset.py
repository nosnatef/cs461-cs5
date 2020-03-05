
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle



def main():

    all_data = grab_json_file('data/filteredSolves.json')

    studies = grab_json_file('data/filteredStudies.json')

    data, data_key_name = new_data(all_data, 'final_iteration')

    studies_data, studies_data_key_name = new_data(studies)

    # for entry in studies_data['study']:
    #     print("*******************************")
    #     for k, v in entry.items():
    #         print("{:<30} = {}".format(k, v))

    # the entry 'id' is where the solveTimes refer to

    studies_solve_data = grab_solvetimes_file('data/solveTimes.txt')

    entry = ['number_of_keep_outs', 'number_of_loads', 'number_of_geometries', 'number_of_keep_ins']
    entry_name = 'four'

    dataset = generate_dataset(studies_data, studies_solve_data, entry = entry, file_path = 'data')

    dataset = np.asarray(dataset)

    write_dataset_to_file(dataset, entry_name)


    # To check if pickle worked properly
    file_name = entry_name + '_dataset.p'
    with open ('data/' + file_name, 'rb') as fp:
        itemlist = pickle.load(fp) 
    
    print(itemlist)
    print(itemlist.shape)


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


def generate_dataset(x_data, y_data, entry, file_path):

    dataset = []

    for x_entry in x_data['study']:
        group = []
        for y_entry in y_data:
            if x_entry['id'] == y_entry[0]:
                for name in entry:
                    if name in x_entry:
                        # print(name)
                        group.append(x_entry[name])
                    # else:
                        # print(name)
                
                group.append(y_entry[1])
        if len(group) > 0:
            dataset.append(group)

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
        entry[1] = (float(entry[1]) / 3600)
        solve_data.append(entry)

    return solve_data


if __name__ == "__main__":
    main()