import os
import pprint
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize

import wandb

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text
from zero_shot import run_cxr_zero_shot, run_zero_shot


def parse_args():
    parser = argparse.ArgumentParser()
    # Chest X-ray 이미지 데이터 경로
    parser.add_argument('--cxr_filepath', type=str, default='./data/cxr.h5',
                        help="Directory to load chest x-ray image data from.")
    # radiology report text(impression section) 경로
    parser.add_argument('--txt_filepath', type=str, default='./data/mimic_impressions.csv',
                        help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    # CLIP 모델에서 사용할 최대 텍스트 문맥 길이
    parser.add_argument('--context_length', type=int, default=77)
    # True일 경우 모델을 무작위로 초기화
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    return args


# 전체 모델 파이프라인
def model_pipeline(config, verbose=0):
    # make the model, data, and optimization problem
    # 모델, 데이터, optiizer 구성
    model, data_loader, device, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose:
        print(model)  # verbose가 켜져 있으면 모델 아키텍처 출력
    return model


# 모델, dataloader, optimizer function 생성
def make(config):
    # random_init이 True일 경우 pre-train된 weight 사용 X
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size,
                                    pretrained=pretrained, column="impression", cuda=config.cuda)
    # CLIP 모델 불러오고, 필요에 따라 pretrained weight 불러옴
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length, cuda=config.cuda)
    model.to(device)
    print(f'----------Model on Device.-----------{device}')

    # make the optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=1e-6)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    return model, data_loader, device, criterion, optimizer


# 모델 훈련
def train(model, loader, device, criterion, optimizer, config):
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir):
        # Create a new folder if not exists
        # 디렉토리가 없으면 생성
        os.makedirs(model_save_dir)

    # Run training
    total_batches = len(loader) * config.epochs  # 전체 batch 수 계산
    example_ct = 0  # number of examples seen 본 예제 수
    batch_ct = 0  # 현재 배치 수
    report_freq = config.log_interval  # loss logging 빈도
    highest_val_auc = 0  # save highest mean auc

    for epoch in range(config.epochs):
        # Batch당 loss를 누적 저장
        running_loss = 0.0  # running loss over batch
        for data in tqdm(loader):
            # get the images
            images = data['img']

            texts = data['txt']
            texts = preprocess_text(texts, model)  # 텍스트 preprocess

            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer, config)
            example_ct += len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0

            if (batch_ct % config.save_interval) == 0:
                model_path = os.path.join(model_save_dir, "checkpoint_{batch_ct}.pt".format(
                    batch_ct=str(batch_ct),
                ))
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)


# 단일 배치에 대해 훈련 실행
def train_batch(images, texts, model, device, criterion, optimizer, config):
    images, texts = images.to(device), texts.to(device)

    # Forward pass ➡
    logits_per_image, logits_per_text = model(images, texts)

    # Create labels
    # Batch size 만큼의 정답 label 생성
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)

    # raise Exception("stop")
    # Compute loss
    # 이미지와 텍스트 logit에 대해 각각 계산 후 평균
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2  # avg. img and txt loss

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # apply gradient clipping when the optmizer is AdamW
    if config.optimizer == 'adam':
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "loss": loss})
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def save(model, path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="CheXzero", name=args.model_name)
    model = model_pipeline(args)


