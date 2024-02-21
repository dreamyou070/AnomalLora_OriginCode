import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
def main(args) :
    file_dir = args.file_dir
    folder, file_name = os.path.split(file_dir)
    file_name, ext = os.path.splitext(file_name)
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    title_list = lines[0].split(",")
    print(title_list)
    title_1 = title_list[1].strip()
    title_1 = title_1.replace("'","")

    if len(title_list) > 2 :
        title_2 = title_list[2].strip()
        title_2 = title_2.replace("'","")

    x_list, y1_list, y2_list = [], [], []
    for line in lines[1:] :
        value = line.split(",")
        step = value[0].strip()
        normal_dist_max = value[1].strip()

        x_list.append(int(step))
        y1_list.append(float(normal_dist_max))
        if len(title_list) > 2:
            down_dimed_normal_dist_mas = value[2].strip()
            y2_list.append(float(down_dimed_normal_dist_mas))

    print(f' step 2. make graph')
    plt.figure(figsize=(10, 5))
    plt.plot(x_list, y1_list, label='normal_dist_max')
    plt.title(title_1)
    plt.savefig(os.path.join(folder, f'{file_name}_{title_1}.png'))

    if len(title_list) > 2:
        plt.figure(figsize=(10, 5))
        plt.plot(x_list, y2_list, label='down_dimed_normal_dist_mas')
        plt.title(title_2)
        name = f"{file_name}_{title_2}"
        name = name.replace("'","")

        dir2 = os.path.join(folder, f'{name}.png')
        print(f' save graph at {dir2}')
        plt.savefig(dir2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="files/normalizing_mahal_feat_log.txt")
    args = parser.parse_args()
    main(args)