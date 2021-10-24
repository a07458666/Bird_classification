def readClassIdx(args):
    class_to_idx = {}
    with open(args.classes_path) as f:
        for line in f.readlines():
            label_num = line.split(".")[0]
            label_str = line.split(".")[1][:-1]
            class_to_idx[int(label_num) - 1] = label_str
    return class_to_idx


def readTrainImages(args):
    data_list = []
    with open(args.training_labels_path) as f:
        for line in f.readlines():
            file_name = line.split(" ")[0]
            label_num = int(line.split(" ")[1].split(".")[0]) - 1
            label_str = line.split(" ")[1].split(".")[1][:-1]
            data_list.append([file_name, label_num, label_str])
    return data_list


def readTestImagesPath(args):
    with open(args.test_filename_path) as f:
        test_images = [x.strip() for x in f.readlines()]
    return test_images


def splitDataList(data_list, train_rate=0.8, val_rate=0.1):
    train_data_list = data_list[: int(len(data_list) * train_rate)]
    val_data_list = data_list[
        int(len(data_list) * train_rate) : int(
            len(data_list) * (train_rate + val_rate)
        )
    ]
    test_data_list = data_list[int(len(data_list) * (train_rate + val_rate)) :]
    return train_data_list, val_data_list, test_data_list
