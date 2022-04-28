# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import criteria
import networks
from utils import dataset

torch.backends.cudnn.benchmark = True


def check_paths(args):
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        new_log_dir = os.path.join(args.log_dir,
                                   time.ctime().replace(" ", "-"))
        args.log_dir = new_log_dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args, device):

    def train_epoch(epoch, steps):
        model.train()
        loss_avg = 0.0
        acc_avg = 0.0
        counter = 0
        train_loader_iter = iter(train_loader)
        for _ in trange(len(train_loader_iter)):
            steps += 1
            counter += 1
            images, targets = next(train_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            loss = criterion(model_output, targets)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, targets)
            acc_avg += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Train Loss", loss.item(), steps)
            writer.add_scalar("Train Acc", acc.item(), steps)
        print("Epoch {}, Total Iter: {}, Train Avg Loss: {:.6f}".format(
            epoch, counter, loss_avg / float(counter)))

        return steps

    def validate_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        counter = 0
        val_loader_iter = iter(val_loader)
        for _ in trange(len(val_loader_iter)):
            counter += 1
            images, targets = next(val_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            loss = criterion(model_output, targets)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, targets)
            acc_avg += acc.item()
        writer.add_scalar("Valid Avg Loss", loss_avg / float(counter), steps)
        writer.add_scalar("Valid Avg Acc", acc_avg / float(counter), steps)
        print("Epoch {}, Valid Avg Loss: {:.6f}, Valid Avg Acc: {:.4f}".format(
            epoch, loss_avg / float(counter), acc_avg / float(counter)))

    def test_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        counter = 0
        test_loader_iter = iter(test_loader)
        for _ in trange(len(test_loader_iter)):
            counter += 1
            images, targets = next(test_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            loss = criterion(model_output, targets)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, targets)
            acc_avg += acc.item()
        writer.add_scalar("Test Avg Loss", loss_avg / float(counter), steps)
        writer.add_scalar("Test Avg Acc", acc_avg / float(counter), steps)
        print("Epoch {}, Test  Avg Loss: {:.6f}, Test  Avg Acc: {:.4f}".format(
            epoch, loss_avg / float(counter), acc_avg / float(counter)))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = networks.CoPINet()
    if args.cuda and args.multigpu and torch.cuda.device_count() > 1:
        print("Running the model on {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        args.lr)

    train_set = dataset(args.dataset, "train", args.img_size)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_set = dataset(args.dataset, "val", args.img_size)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_set = dataset(args.dataset, "test", args.img_size)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    criterion = criteria.contrast_loss

    writer = SummaryWriter(args.log_dir)

    total_steps = 0

    for epoch in range(args.epochs):
        total_steps = train_epoch(epoch, total_steps)
        with torch.no_grad():
            validate_epoch(epoch, total_steps)
            test_epoch(epoch, total_steps)

        # save checkpoint
        model.eval().cpu()
        ckpt_model_name = "epoch_{}_batch_{}_seed_{}_lr_{}.pth".format(
            epoch, args.batch_size, args.seed, args.lr)
        ckpt_file_path = os.path.join(args.checkpoint_dir, ckpt_model_name)
        torch.save(model.state_dict(), ckpt_file_path)
        model.to(device)

    # save final model
    model.eval().cpu()
    save_model_name = "Final_epoch_{}_batch_{}_seed_{}_lr_{}.pth".format(
        epoch, args.batch_size, args.seed, args.lr)
    save_file_path = os.path.join(args.save_dir, save_model_name)
    torch.save(model.state_dict(), save_file_path)

    print("Done. Model saved.")


def test(args, device):

    def test_epoch():
        model.eval()
        correct_avg = 0.0
        test_loader_iter = iter(test_loader)
        for _ in range(len(test_loader_iter)):
            images, targets = next(test_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            correct_num = criteria.calculate_correct(model_output, targets)
            correct_avg += correct_num.item()
        print("Test Avg Acc: {:.4f}".format(correct_avg /
                                            float(test_set_size)))

    model = networks.CoPINet()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    subfolders = [
        os.path.join(args.dataset, folder) for folder in [
            "center_single", "distribute_four", "distribute_nine",
            "in_center_single_out_center_single",
            "in_distribute_four_out_center_single",
            "left_center_single_right_center_single",
            "up_center_single_down_center_single"
        ]
    ]
    subfolders.append(os.path.join(args.dataset, "*"))
    for folder in subfolders:
        print("Evaluating on {}".format(folder))
        test_set = dataset(folder, "test", args.img_size, test=True)
        test_set_size = len(test_set)
        test_loader = DataLoader(test_set,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        with torch.no_grad():
            test_epoch()


def main():
    main_arg_parser = argparse.ArgumentParser(description="CoPINet")
    subparsers = main_arg_parser.add_subparsers(title="subcommands",
                                                dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training")
    train_arg_parser.add_argument("--epochs",
                                  type=int,
                                  default=200,
                                  help="the number of training epochs")
    train_arg_parser.add_argument("--batch-size",
                                  type=int,
                                  default=32,
                                  help="size of batch")
    train_arg_parser.add_argument("--seed",
                                  type=int,
                                  default=1234,
                                  help="random number seed")
    train_arg_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="device index for GPU; if GPU unavailable, leave it as default")
    train_arg_parser.add_argument("--num-workers",
                                  type=int,
                                  default=16,
                                  help="number of workers for data loader")
    train_arg_parser.add_argument(
        "--dataset",
        type=str,
        default="/home/chizhang/Datasets/RAVEN-10000/",
        help="dataset path")
    train_arg_parser.add_argument("--checkpoint-dir",
                                  type=str,
                                  default="./experiments/ckpt/",
                                  help="checkpoint save path")
    train_arg_parser.add_argument("--save-dir",
                                  type=str,
                                  default="./experiments/save/",
                                  help="final model save path")
    train_arg_parser.add_argument("--log-dir",
                                  type=str,
                                  default="./experiments/log/",
                                  help="log save path")
    train_arg_parser.add_argument("--img-size",
                                  type=int,
                                  default=80,
                                  help="image size for training")
    train_arg_parser.add_argument("--lr",
                                  type=float,
                                  default=1e-3,
                                  help="learning rate")
    train_arg_parser.add_argument("--multigpu",
                                  type=int,
                                  default=0,
                                  help="whether to use multi gpu")

    test_arg_parser = subparsers.add_parser("test", help="parser for testing")
    test_arg_parser.add_argument("--batch-size",
                                 type=int,
                                 default=32,
                                 help="size of batch")
    test_arg_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="device index for GPU; if GPU unavailable, leave it as default")
    test_arg_parser.add_argument("--num-workers",
                                 type=int,
                                 default=16,
                                 help="number of workers for data loader")
    test_arg_parser.add_argument("--dataset",
                                 type=str,
                                 default="/home/chizhang/Datasets/RAVEN-10000",
                                 help="dataset path")
    test_arg_parser.add_argument("--model-path",
                                 type=str,
                                 required=True,
                                 help="path to a trained model")
    test_arg_parser.add_argument("--img-size",
                                 type=int,
                                 default=80,
                                 help="image size for training")

    args = main_arg_parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda:{}".format(args.device) if args.cuda else "cpu")

    if args.subcommand is None:
        print("ERROR: Specify train or test")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args, device)
    elif args.subcommand == "test":
        test(args, device)
    else:
        print("ERROR: Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()
