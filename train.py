import os
import argparse
import torch
import numpy as np
import math

from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau
from matplotlib import pyplot as plt
from IPython.display import clear_output
from torch.cuda.amp import GradScaler, autocast

from src.data_loading.data_loader import BirdImageLoader
from src.txt_loading.txt_loader import (
    readClassIdx,
    readTrainImages,
    splitDataList,
)
from src.loss_functions.CrossEntropyLS import CrossEntropyLS


def main(args):
    from torch.utils.tensorboard import SummaryWriter

    writer = create_writer(args)
    return
    device = checkGPU()
    class_to_idx = readClassIdx(args)
    data_list = readTrainImages(args)
    train_data_list, val_data_list, _ = splitDataList(data_list)
    model = create_model(args, device)
    train_loader, val_loader = create_dataloader(
        args, train_data_list, val_data_list, class_to_idx
    )
    checkOutputDirectoryAndCreate(args)
    train(args, model, train_loader, val_loader, writer, device)


def checkOutputDirectoryAndCreate(args):
    if not os.path.exists(args.output_foloder):
        os.makedirs(args.output_foloder)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def checkGPU():
    print("torch version:" + torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Available GPUs: ", end="")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i), end=" ")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")
    return device


def update_loss_hist(args, train_list, val_list, name="result"):
    clear_output(wait=True)
    plt.plot(train_list)
    plt.plot(val_list)
    plt.title(name)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="center right")
    plt.savefig("{}/{}.png".format(args.output_foloder, name))
    # plt.show()


def create_model(args, device):
    import timm

    backbone = timm.create_model(
        "vit_base_patch16_224_miil_in21k", pretrained=True
    )

    if args.pretrain_model_path != '':
        backbone = torch.load(args.pretrain_model_path).to(device)
        set_parameter_requires_grad(backbone, False)

    projector = nn.Sequential(
        nn.Linear(11221, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 200),
    )
    model = nn.Sequential(backbone, projector).to(device)
    return model


def create_dataloader(args, train_data_list, val_data_list, class_to_idx):
    from src.helper_functions.augmentations import (
        get_aug_trnsform,
        get_eval_trnsform,
    )

    trans_aug = get_eval_trnsform()
    trans_eval = get_eval_trnsform()
    dataset_train = BirdImageLoader(
        args.data_path, train_data_list, class_to_idx, transform=trans_aug
    )
    dataset_val = BirdImageLoader(
        args.data_path, val_data_list, class_to_idx, transform=trans_eval
    )

    train_loader = DataLoader(
        dataset_train,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset_val,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

    print("class_to_idx ", len(class_to_idx))
    print("train len", dataset_train.__len__())
    print("val len", dataset_val.__len__())
    return train_loader, val_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top
    predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def pass_epoch(
    model, loader, model_optimizer, loss_fn, scaler, device, mode="Train"
):
    loss = 0
    acc_top1 = 0
    acc_top5 = 0

    for i_batch, image_batch in tqdm(enumerate(loader)):
        x, y = image_batch[0].to(device), image_batch[1].to(device)
        if mode == "Train":
            model.train()
        elif mode == "Eval":
            model.eval()
        else:
            print("error model mode!")
        y_pred = model(x)

        loss_batch = loss_fn(y_pred, y)
        loss_batch_acc_top = accuracy(y_pred, y, topk=(1, 5))

        if mode == "Train":
            model_optimizer.zero_grad()
            scaler.scale(loss_batch).backward()
            scaler.step(model_optimizer)
            scaler.update()
            model_optimizer.step()

        loss += loss_batch.detach().cpu()
        acc_top1 += loss_batch_acc_top[0]
        acc_top5 += loss_batch_acc_top[1]

    loss /= i_batch + 1
    acc_top1 /= i_batch + 1
    acc_top5 /= i_batch + 1
    return loss, acc_top1, acc_top5


def create_writer(args):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter("runs/" + args.output_foloder)
    for key in vars(args):
        msg = "{} = {}".format(key, vars(args)[key])
        print(msg)
        writer.add_text("Remark", msg, 0)
    # writer.add_text('Remark', 'batch_size = {}'.format(args.batch_size) , 0)
    # writer.add_text('Remark', 'test!!!' , 0)
    writer.flush()
    writer.close()
    return writer


def train(args, model, train_loader, val_loader, writer, device):
    train_loss_history = []
    train_acc_top1_history = []
    train_acc_top5_history = []
    val_loss_history = []
    val_acc_top1_history = []
    val_acc_top5_history = []
    model_optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model_scheduler = ReduceLROnPlateau(model_optimizer, "min")
    torch.save(model, "{}/checkpoint.pth.tar".format(args.output_foloder))
    loss_fn = CrossEntropyLS(args.label_smooth)
    scaler = GradScaler()
    stop = 0
    min_val_loss = math.inf
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        print("-" * 10)
        train_loss, train_acc_top1, train_acc_top5 = pass_epoch(
            model,
            train_loader,
            model_optimizer,
            loss_fn,
            scaler,
            device,
            "Train",
        )
        with torch.no_grad():
            val_loss, val_acc_top1, val_acc_top5 = pass_epoch(
                model,
                val_loader,
                model_optimizer,
                loss_fn,
                scaler,
                device,
                "Eval",
            )
        model_scheduler.step(val_loss)

        writer.add_scalars(
            "loss", {"train": train_loss, "val": val_loss}, epoch
        )
        writer.add_scalars(
            "top1", {"train": train_acc_top1, "val": val_acc_top1}, epoch
        )
        writer.add_scalars(
            "top5", {"train": train_acc_top5, "val": val_acc_top5}, epoch
        )
        writer.flush()

        train_loss_history.append(train_loss)
        train_acc_top1_history.append(train_acc_top1)
        train_acc_top5_history.append(train_acc_top5)

        val_loss_history.append(val_loss)
        val_acc_top1_history.append(val_acc_top1)
        val_acc_top5_history.append(val_acc_top5)

        update_loss_hist(args, train_loss_history, val_loss_history, "Loss")
        update_loss_hist(
            args, train_acc_top5_history, val_acc_top5_history, "Top5"
        )
        update_loss_hist(
            args, train_acc_top1_history, val_acc_top1_history, "Top1"
        )
        if val_loss <= min_val_loss:
            val_loss = min_val_loss
            torch.save(
                model,
                "{}/checkpoint_{:04d}.pth.tar".format(
                    args.output_foloder, epoch + 1
                ),
            )
        else:
            stop += 1
            if stop > 5:
                print("early stopping")
                break
    torch.save(model, "{}/checkpoint.pth.tar".format(args.output_foloder))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="310551010 train bird")
    parser.add_argument(
        "--data_path", type=str, default="../../dataset/bird_datasets/train"
    )
    parser.add_argument(
        "--classes_path",
        type=str,
        default="../../dataset/bird_datasets/classes.txt",
    )
    parser.add_argument(
        "--training_labels_path",
        type=str,
        default="../../dataset/bird_datasets/training_labels.txt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--label_smooth",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--pretrain_model_path",
        type=str,
        default="model/model_bird_vic_simsiam_pretrain/checkpoint.pth.tar",
    )
    parser.add_argument(
        "--output_foloder",
        type=str,
        default="model/model_test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    main(args)
