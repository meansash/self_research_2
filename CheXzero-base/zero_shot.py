import subprocess
from pathlib import Path

import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
from copy import deepcopy
import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

import clip
from model import CLIP
from eval import evaluate, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from torchvision.datasets import ImageFolder

# TODO : Change CXR_FILEPATH, FINAL_LABEL_PATH
CXR_FILEPATH = './data/test_cxr.h5'
FINAL_LABEL_PATH = './data/final_paths.csv'


# class ChestXray14Dataset(data.Dataset):
#     def __init__(self, transform=None):
#         super().__init__()
#         testdir = os.path.join("/home/meansash/shared/hdd_ext/nvme1/public/medical/classification/chest/ChestXray14/images")
#         testset = ImageFolder(testdir, transform=transform)
#
#     def __len__(self):
#         return len(self.testset)
#
#     def __getitem__(self, idx):
#


# h5 파일로부터 데이터를 읽어들이는 데이터셋 클래스 정의
class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(
            self,
            img_path: str,
            transform=None,
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']  # h5 파일의 'cxr' 데이터셋을 불러옴
        self.transform = transform  # image transform

    def __len__(self):
        return len(self.img_dset)  # dataset length 반ㅂ환

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx]  # np array, (320, 320)
        img = np.expand_dims(img, axis=0)  # 차원 확장 (채널 추가)
        img = np.repeat(img, 3, axis=0)  # 흑백 이미지를 RGB로 변환
        img = torch.from_numpy(img)  # torch, (320, 320)

        if self.transform:
            img = self.transform(img)  # transform 적용
        sample = {'img': img}

        return sample


def load_clip(model_path, pretrained=False, context_length=77):
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cpu")
    if pretrained is False:
        # use new model params
        params = {
            'embed_dim': 768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': context_length,
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }
        # CLIP model initialize
        model = CLIP(**params)
    else:
        # pretrained CLIP model 로드
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    try:
        # model weight load
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print("Argument error. Set pretrained = True.", sys.exc_info()[0])
        raise
    return model


def zeroshot_classifier(classnames, templates, model, context_length=77):
    """
    FUNCTION: zeroshot_classifier
    -------------------------------------
    This function outputs the weights for each of the classes based on the
    output of the trained clip model text transformer.

    args:
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis',...]).
    * templates - Python list of phrases that will be independently tested as input to the clip model.
    * model - Pytorch model, full trained clip model.
    * context_length (optional) - int, max number of tokens of text inputted into the model.

    Returns PyTorch Tensor, output of the text encoder given templates.
    """
    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            # template에 class name 적용
            texts = [template.format(classname) for template in templates]  # format with class
            # text tokenizing
            texts = clip.tokenize(texts, context_length=context_length)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates
            class_embedding = class_embeddings.mean(dim=0)
            # norm over new averaged templates
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


# Test-Time Augmentation(TTA)를 활용한 zero-shot classifier
def zeroshot_classifier_with_tta(model, negative=False):
    """
    FUNCTION: zeroshot_classifier
    -------------------------------------
    This function outputs the weights for each of the classes based on the
    output of the trained clip model text transformer.

    args:
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis',...]).
    * templates - Python list of phrases that will be independently tested as input to the clip model.
    * model - Pytorch model, full trained clip model.
    * context_length (optional) - int, max number of tokens of text inputted into the model.

    Returns PyTorch Tensor, output of the text encoder given templates.
    """
    with torch.no_grad():
        text_features = []
        if negative:
            prompts = model.neg_prompt_learner()
            tokenized_prompts = model.neg_prompt_learner.tokenized_prompts

        else:
            prompts = model.prompt_learner()
            tokenized_prompts = model.prompt_learner.tokenized_prompts
        t_features = model.text_encoder(prompts, tokenized_prompts)

        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)


# prediction 함수 정의
def predict(loader, model, zeroshot_weights, softmax_eval=True, verbose=0):
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model
    and computes the cosine similarities between the images
    and the text embeddings.

    args:
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.

    Returns numpy array, predictions on all test data samples.
    """
    y_pred = []
    mean = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img']
            # predict
            image_features = model.encode_image(images)  # image embedding 생성
            if i < 100:
                # save image jpg
                image = images.cpu().numpy()
                image = np.squeeze(image, axis=0)
                image = np.transpose(image, (1, 2, 0))
                image = (image - image.min()) / (image.max() - image.min())  # normalize
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                # TODO : save path 변경
                image.save("/home/meansash/private/zero_shot_medical/CheXzero_base/temp/{}.jpg".format(i))
            image_features /= image_features.norm(dim=-1, keepdim=True)  # (1, 768) # normalize
            mean.append(image_features.mean(dim=-1))
            # print(zeroshot_weights.shape)
            # obtain logits
            logits = image_features @ zeroshot_weights  # (1, num_classes) # 클래스별 logit 계산
            logits = np.squeeze(logits.numpy(), axis=0)  # (num_classes,)

            if softmax_eval is False:
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits)

            y_pred.append(logits)

            if verbose:
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())

                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())

    print('features mean: ', torch.cat(mean).mean())
    print('features var: ', torch.cat(mean).var())
    y_pred = np.array(y_pred)
    return np.array(y_pred)


# Zero-shot prediction을 위한 TTA 적용 예측 함수 정의
def predict_tta(loader, model, cxr_labels, tta_steps, softmax_eval=True, verbose=0, optimizer=None, scaler=None,
                optim_state=None, device=None, negative=False):
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model
    and computes the cosine similarities between the images
    and the text embeddings.

    args:
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.

    Returns numpy array, predictions on all test data samples.
    """
    y_pred = []
    mean = []
    for i, (data, idx) in enumerate(tqdm(loader)):
        images = data['img']
        # print("idx: ", idx)
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
        images = images.to(torch.float32).to(device, non_blocking=True)

        if tta_steps > 0:
            with torch.no_grad():
                model.reset()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, images, optimizer, scaler, tta_steps)

            image = images[0].unsqueeze(0)
        else:
            image = images
        # predict
        with torch.no_grad():
            model.eval()
            image_features = model.visual(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)  # (1, 768)
            mean.append(image_features.mean(dim=-1))
            zeroshot_weights = zeroshot_classifier_with_tta(model, negative=negative)
            # obtain logits
            logits = image_features @ zeroshot_weights.t()  # (1, num_classes)
            # print("logits: ", logits.shape)
            logits = np.squeeze(logits.cpu().numpy(), axis=0)  # (num_classes,)

            # logits = logits.cpu().numpy()
            if softmax_eval is False:
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits)

            y_pred.append(logits)

            if verbose:
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())

                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())

    print('features mean: ', torch.cat(mean).mean())
    print('features var: ', torch.cat(mean).var())
    # print(y_pred)
    y_pred = np.array(y_pred)
    # print(y_pred.shape)

    return np.array(y_pred)


def run_single_prediction(cxr_labels, template, model, loader, softmax_eval=True, context_length=77):
    """
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}").

    args:
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model.
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation
        * context_length (optional) - int, max number of tokens of text inputted into the model.

    Returns list, predictions from the given template.
    """
    cxr_phrase = [template]
    # print(cxr_labels)
    zeroshot_weights = zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length)

    y_pred = predict(loader, model, zeroshot_weights, softmax_eval=softmax_eval)
    return y_pred


def process_alt_labels(alt_labels_dict, cxr_labels):
    """
        Process alt labels and return relevant info. If `alt_labels_dict` is
        None, return None.

    Returns:
    * alt_label_list : list
        List of all alternative labels
    * alt_label_idx_map : dict
        Maps alt label to idx of original label in cxr_labels
        Needed to access correct column during evaluation

    """

    if alt_labels_dict is None:
        return None, None

    def get_inverse_labels(labels_alt_map: dict):
        """
        Returns dict mapping alternative label back to actual label.
        Used for reference during evaluation.
        """
        inverse_labels_dict = {}
        for main in labels_alt_map:
            inverse_labels_dict[main] = main  # adds self to list of alt labels
            for alt in labels_alt_map[main]:
                inverse_labels_dict[alt] = main
        return inverse_labels_dict

    inv_labels_dict = get_inverse_labels(alt_labels_dict)
    alt_label_list = [w for w in inv_labels_dict.keys()]

    # create index map
    index_map = dict()
    for i, label in enumerate(cxr_labels):
        index_map[label] = i

    # make map to go from alt label directly to index
    alt_label_idx_map = dict()
    for alt_label in alt_label_list:
        alt_label_idx_map[alt_label] = index_map[inv_labels_dict[alt_label]]

    return alt_label_list, alt_label_idx_map


def get_pair_template(label: str):
    '''
    Return a specific prompt template based on the input label

    '''
    # Default prompt
    # pos = "{}"
    # neg = "No {}"

    # Xplainer prompt
    # pos = "There is observation indicating {}"
    # neg = "There is no observation indicating {}"

    # ratio = random.random()

    # Chat-GPT prompt
    # pos = "The X-ray shows some suspicious areas related to {}"
    # neg = "The X-ray reveals no evidence of {}"
    #

    # Custom prompt
    # ours-1
    if label == "Atelectasis":
        pos = "The X-ray shows {} in patient's lung"
        neg = "The X-ray shows no {} in patient's lung"
    elif label == "Cardiomegaly":
        pos = "The X-ray shows an enlarged heart which means {}"
        neg = "The X-ray shows no enlarged heart which means {}"
    elif label == "Consolidation":
        pos = "The X-ray shows {} in patient's lung"
        neg = "The X-ray shows no {} in patient's lung"
    elif label == "Edema":
        pos = "The X-ray shows pulmonary {} at patient's lung"
        neg = "The X-ray shows no pulmonary {} at patient's lung"
    elif label == "Enlarged Cardiomediastinum":
        pos = "The X-ray shows {} at patient's chest"
        neg = "The X-ray shows no {} at patient's chest"
    elif label == "Fracture":
        pos = "The X-ray shows {} at patient's rib"
        neg = "The X-ray shows no {} at patient's rib"
    elif label == "Lung Lesion":
        pos = "The X-ray shows a {} at patient's lung"
        neg = "The X-ray shows no {} at patient's lung"
    elif label == "Lung Opacity":
        pos = "The X-ray shows {} in patient's lung"
        neg = "The X-ray shows no {} in patient's lung"
    elif label == "No Finding":
        pos = "{}"
        neg = "There is something in X-ray"
    elif label == "Pleural Effusion":
        pos = "The X-ray shows {} in patient's lung"
        neg = "The X-ray shows no {} in patient's lung"
    elif label == "Pleural Other":
        pos = "The X-ray shows {}"
        neg = "The X-ray shows no {}"
    elif label == "Pneumonia":
        pos = "The X-ray shows {} in patient's lung"
        neg = "The X-ray shows no {} in patient's lung"
    elif label == "Pneumothorax":
        pos = "The X-ray shows {} in patient's lung"
        neg = "The X-ray shows no {} in patient's lung"
    else:
        # Support Devices
        pos = "The X-ray shows {} in patient's body"
        neg = "The X-ray shows no {} in patient's body"

    # ours-2
    # if label == "Atelectasis":
    #     pos = "The X-ray shows signs of {} in patient's lung"
    #     neg = "The X-ray shows no sings of {} in patient's lung"
    # elif label == "Cardiomegaly":
    #     pos = "The X-ray shows shows an enlarged heart which means {}"
    #     neg = "The X-ray shows doesn't shows enlarged heart which means {}"
    # elif label == "Consolidation":
    #     pos = "The X-ray shows signs of{} in patient's lung"
    #     neg = "The X-ray shows no signs of{} in patient's lung"
    # elif label == "Edema":
    #     pos = "The X-ray shows pulmonary {} at patient's lung"
    #     neg = "The X-ray doesn't shows pulmonary {} at patient's lung"
    # elif label == "Enlarged Cardiomediastinum":
    #     pos = "The X-ray reveals an {} at patient's chest"
    #     neg = "The X-ray doesn't reveals an {} at patient's chest"
    # elif label == "Fracture":
    #     pos = "The X-ray reveals a {} at patient's rib"
    #     neg = "The X-ray doesn't reveals an {} at patient's rib"
    # elif label == "Lung Lesion":
    #     pos = "The X-ray shows a {} at patient's lung"
    #     neg = "The X-ray doesn't shows a {} at patient's lung"
    # elif label == "Lung Opacity":
    #     pos = "There is an {} in patient's lung"
    #     neg = "There isn't an {} in patient's lung"
    # elif label == "No Finding":
    #     pos = "{}"
    #     neg = "There is something in X-ray"
    # elif label == "Pleural Effusion":
    #     pos = "The X-ray reveals a {} in patient's lung"
    #     neg = "The X-ray doesn't reveals a {} in patient's lung"
    # elif label == "Pleural Other":
    #     pos = "The X-ray shows signs of a{}"
    #     neg = "The X-ray doesn't shows signs of a {}"
    # elif label == "Pneumonia":
    #     pos = "The X-ray shows evidence of {} in patient's lung"
    #     neg = "The X-ray doesn't shows evidence of {} in patient's lung"
    # elif label == "Pneumothorax":
    #     pos = "The X-ray reveals a {} in patient's lung"
    #     neg = "The X-ray doesn't reveals a {} in patient's lung"
    # else:
    #     # Support Devices
    #     pos = "The X-ray shows {} in patient's body"
    #     neg = "The X-ray doesn't shows {} in patient's body"
    #
    # ours-3
    # if label == "Atelectasis":
    #     pos = "The X-ray shows {} in patient's lung"
    #     neg = "The X-ray doesn't shows {} in patient's lung"
    # elif label == "Cardiomegaly":
    #     pos = "The X-ray shows an enlarged heart which means {}"
    #     neg = "The X-ray doesn't shows enlarged heart which means {}"
    # elif label == "Consolidation":
    #     pos = "The X-ray shows {} in patient's lung"
    #     neg = "The X-ray doesn't shows {} in patient's lung"
    # elif label == "Edema":
    #     pos = "The X-ray shows pulmonary {} at patient's lung"
    #     neg = "The X-ray doesn't shows pulmonary {} at patient's lung"
    # elif label == "Enlarged Cardiomediastinum":
    #     pos = "The X-ray shows {} at patient's chest"
    #     neg = "The X-ray doesn't shows {} at patient's chest"
    # elif label == "Fracture":
    #     pos = "The X-ray shows {} at patient's rib"
    #     neg = "The X-ray doesn't shows {} at patient's rib"
    # elif label == "Lung Lesion":
    #     pos = "The X-ray shows a {} at patient's lung"
    #     neg = "The X-ray doesn't shows {} at patient's lung"
    # elif label == "Lung Opacity":
    #     pos = "The X-ray shows {} in patient's lung"
    #     neg = "The X-ray doesn't shows {} in patient's lung"
    # elif label == "No Finding":
    #     pos = "{}"
    #     neg = "There is something in X-ray"
    # elif label == "Pleural Effusion":
    #     pos = "The X-ray shows {} in patient's lung"
    #     neg = "The X-ray doesn't shows {} in patient's lung"
    # elif label == "Pleural Other":
    #     pos = "The X-ray shows {}"
    #     neg = "The X-ray doesn't shows {}"
    # elif label == "Pneumonia":
    #     pos = "The X-ray shows {} in patient's lung"
    #     neg = "The X-ray doesn't shows {} in patient's lung"
    # elif label == "Pneumothorax":
    #     pos = "The X-ray shows {} in patient's lung"
    #     neg = "The X-ray doesn't shows {} in patient's lung"
    # else:
    #     # Support Devices
    #     pos = "The X-ray shows {} in patient's body"
    #     neg = "The X-raydoesn't shows {} in patient's body"

    return (pos, neg)


def select_confident_samples(logits, top):
    print(f"logit shape: {logits.shape}")

    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    print(f"batch entropy: {batch_entropy}")
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    print(torch.argsort(batch_entropy, descending=False))
    print(f"selected_idx: {idx}")
    return logits[idx], idx


def avg_entropy(outputs, negative=False):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler, tta_steps):
    selected_idx = None
    selection_p = 0.3

    # print("--------------tuning start---------------")
    for i in range(tta_steps):

        output = model(inputs)

        if selected_idx is not None:
            output = output[selected_idx]

        else:
            output, selected_idx = select_confident_samples(output, selection_p)

        loss = avg_entropy(output)

        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

    return


def tta_single_prediction(cxr_labels, model, loader, tta_steps, softmax_eval=True, context_length=77, device=None,
                          negative=False):
    """ TODO: change this notation to tta
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}").

    args:
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model.
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation
        * context_length (optional) - int, max number of tokens of text inputted into the model.

    Returns list, predictions from the given template.

    """
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.Adam(trainable_param, 0.0001)
    scaler = None
    optim_state = deepcopy(optimizer.state_dict())

    y_pred = predict_tta(loader, model, cxr_labels, tta_steps, softmax_eval=softmax_eval, optimizer=optimizer,
                         scaler=scaler, optim_state=optim_state, device=device, negative=negative)
    return y_pred


def tta_softmax_eval(model, loader, tta_steps, eval_labels: list, context_length: int = 77, device='cuda'):
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """

    pos_pred = tta_single_prediction(eval_labels, model, loader, tta_steps,
                                     softmax_eval=True, context_length=context_length, device=device)
    neg_pred = tta_single_prediction(eval_labels, model, loader, tta_steps,
                                     softmax_eval=True, context_length=context_length, device=device, negative=True)

    # print("pos_pred: ", pos_pred[0])
    # print("neg_pred: ", neg_pred[0])

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred

    # print(y_pred[0])
    return y_pred


def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, context_length: int = 77):
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
    # get pos and neg phrases
    # pos , neg = get_pair_template()
    #
    # #get false label
    #
    # # get pos and neg predictions, (num_samples, num_classes)
    # pos_pred = run_single_prediction(eval_labels, pos, model, loader,
    #                                  softmax_eval=True, context_length=context_length)
    # neg_pred = run_single_prediction(eval_labels, neg, model, loader,
    #                                  softmax_eval=True, context_length=context_length)
    #
    # # compute probabilities with softmax
    # sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    # # print("pos_pred: ", pos_pred[0])
    # # print("neg_pred: ", neg_pred[0])
    # y_pred = np.exp(pos_pred) / sum_pred
    # return y_pred

    all_preds = []
    for label in eval_labels:
        pos, neg = get_pair_template(label)
        pos_pred = run_single_prediction([label], pos, model, loader, softmax_eval=True, context_length=context_length)
        neg_pred = run_single_prediction([label], neg, model, loader, softmax_eval=True, context_length=context_length)

        sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
        y_pred = np.exp(pos_pred) / sum_pred

        all_preds.append(y_pred)
    return np.hstack(all_preds)


def run_experiment(model, cxr_labels, cxr_templates, loader, y_true, alt_labels_dict=None, softmax_eval=True,
                   context_length=77, use_bootstrap=True):
    '''
    FUNCTION: run_experiment
    ----------------------------------------
    This function runs the zeroshot experiment on each of the templates
    individually, and stores the results in a list.

    args:
        * model - PyTorch model, trained clip model
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * cxr_templates - list, templates to input into model. If softmax_eval is True,
        this should be a list of tuples, where each tuple is a +/- pair
        * loader - PyTorch data loader, loads in cxr images
        * y_true - list, ground truth labels for test dataset
        * softmax_eval (optional) - bool, if True, will evaluate results through softmax of pos vs. neg samples.
        * context_length - int, max number of tokens of text inputted into the model.
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling

    Returns a list of results from the experiment.
    '''

    alt_label_list, alt_label_idx_map = process_alt_labels(alt_labels_dict, cxr_labels)
    if alt_label_list is not None:
        eval_labels = alt_label_list
    else:
        eval_labels = cxr_labels

    results = []
    for template in cxr_templates:
        print('Phrase being used: ', template)

        try:
            if softmax_eval:
                y_pred = run_softmax_eval(model, loader, eval_labels, template, context_length=context_length)

            else:
                # get single prediction
                y_pred = run_single_prediction(eval_labels, template, model, loader,
                                               softmax_eval=softmax_eval, context_length=context_length)
        #             print("y_pred: ", y_pred)
        except:
            print("Argument error. Make sure cxr_templates is proper format.", sys.exc_info()[0])
            raise

        # evaluate
        if use_bootstrap:
            # compute bootstrap stats
            boot_stats = bootstrap(y_pred, y_true, eval_labels, label_idx_map=alt_label_idx_map)
            results.append(boot_stats)  # each template has a pandas array of samples
        else:
            stats = evaluate(y_pred, y_true, eval_labels)
            results.append(stats)

    return results, y_pred


def make_true_labels(
        cxr_true_labels_path: str,
        cxr_labels: List[str],
        cutlabels: bool = True
):
    """
    Loads in data containing the true binary labels
    for each pathology in `cxr_labels` for all samples. This
    is used for evaluation of model performance.

    args:
        * cxr_true_labels_path - str, path to csv containing ground truth labels
        * cxr_labels - List[str], subset of label columns to select from ground truth df
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
            with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns a numpy array of shape (# samples, # labels/pathologies)
        representing the binary ground truth labels for each pathology on each sample.
    """
    # create ground truth labels
    full_labels = pd.read_csv(cxr_true_labels_path)

    # drop if path contains view2 or view3
    full_labels = full_labels[~full_labels['Path'].str.contains('view2')]
    full_labels = full_labels[~full_labels['Path'].str.contains('view3')]

    if cutlabels:
        full_labels = full_labels.loc[:, cxr_labels]
    else:
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true


def make(
        model_path: str,
        cxr_filepath: str,
        pretrained: bool = True,
        context_length: bool = 77,
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels.

    args:
        * model_path - String for directory to the weights of the trained clip model.
        * context_length - int, max number of tokens of text inputted into the model.
        * cxr_filepath - String for path to the chest x-ray images.
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns model, data loader.
    """
    # load model
    model = load_clip(
        model_path=model_path,
        pretrained=pretrained,
        context_length=context_length
    )

    # load data
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    # if using CLIP pretrained model
    if pretrained:
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)

    # create dataset
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)

    return model, loader


## Run the model on the data set using ensembled models
def ensemble_models(
        model_paths: List[str],
        cxr_filepath: str,
        cxr_labels: List[str],
        cache_dir: str = None,
        save_name: str = None,
        tta_steps: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths)  # ensure consistency of
    for path in model_paths:  # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`

        model, loader = custom_clip.get_coop(arch, device, n_ctx=4,
                                             ctx_init="The X-ray shows some suspicious areas related to",
                                             neg_ctx_init="The X-ray reveals no evidence of", chest=path,
                                             tta_steps=tta_steps)
        model = model.to(torch.float32).to(device)

        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None:
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else:
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction

        y_pred = tta_softmax_eval(model, loader, tta_steps, cxr_labels, device)

        predictions.append(y_pred)

    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)

    return predictions, y_pred_avg, y_pred


def run_zero_shot(cxr_labels, cxr_templates, model_path, cxr_filepath, final_label_path, alt_labels_dict: dict = None,
                  softmax_eval=True, context_length=77, pretrained: bool = False, use_bootstrap=True, cutlabels=True):
    """
    FUNCTION: run_zero_shot
    --------------------------------------
    This function is the main function to run the zero-shot pipeline given a dataset,
    labels, templates for those labels, ground truth labels, and config parameters.

    args:
        * cxr_labels - list
            labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
            task can either be a string or a tuple (name of alternative label, name of label in csv)
        * cxr_templates - list, phrases that will be indpendently tested as input to the clip model. If `softmax_eval` is True, this parameter should be a
        list of positive and negative template pairs stored as tuples.
        * model_path - String for directory to the weights of the trained clip model.
        * cxr_filepath - String for path to the chest x-ray images.
        * final_label_path - String for path to ground truth labels.

        * alt_labels_dict (optional) - dict, map cxr_labels to list of alternative labels (i.e. 'Atelectasis': ['lung collapse', 'atelectatic lung', ...])
        * softmax_eval (optional) - bool, if True, will evaluate results through softmax of pos vs. neg samples.
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
        * cutlabels (optional) - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns an array of results per template, each consists of a tuple containing a pandas dataframes
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """

    np.random.seed(97)
    # make the model, data loader, and ground truth labels
    model, loader = make(
        model_path=model_path,
        cxr_filepath=cxr_filepath,
        pretrained=pretrained,
        context_length=context_length
    )

    y_true = make_true_labels(
        cxr_true_labels_path=final_label_path,
        cxr_labels=cxr_labels,
        cutlabels=cutlabels,
    )

    print("y_true: ", y_true)

    y_neg = 1 - y_true  # get negative labels

    # run multiphrase experiment
    results, y_pred = run_experiment(model, cxr_labels, cxr_templates, loader, y_true,
                                     alt_labels_dict=alt_labels_dict, softmax_eval=softmax_eval,
                                     context_length=context_length, use_bootstrap=use_bootstrap)
    return results, y_pred


def run_cxr_zero_shot(model_path, context_length=77, pretrained=False):
    """
    FUNCTION: run_cxr_zero_shot
    --------------------------------------
    This function runs zero-shot specifically for the cxr dataset.
    The only difference between this function and `run_zero_shot` is that
    this function is already pre-parameterized for the 14 cxr labels evaluated
    using softmax method of positive and negative templates.

    args:
        * model_path - string, filepath of model being evaluated
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling

    Returns an array of labels, and an array of results per template,
    each consists of a tuple containing a pandas dataframes
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    cxr_filepath = '/deep/group/data/med-data/test_cxr.h5'
    final_label_path = '/deep/group/data/med-data/final_paths.csv'

    cxr_labels = ['Atelectasis', 'Cardiomegaly',
                  'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                  'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                  'Pneumothorax', 'Support Devices']

    # templates list of positive and negative template pairs
    cxr_templates = [("{}", "no {}")]

    cxr_results = run_zero_shot(cxr_labels, cxr_templates, model_path, cxr_filepath=cxr_filepath,
                                final_label_path=final_label_path, softmax_eval=True, context_length=context_length,
                                pretrained=pretrained, use_bootstrap=False, cutlabels=True)

    return cxr_labels, cxr_results[0]


def validation_zero_shot(model_path, context_length=77, pretrained=False):
    """
    FUNCTION: validation_zero_shot
    --------------------------------------
    This function uses the CheXpert validation dataset to make predictions
    on an alternative task (ap/pa, sex) in order to tune hyperparameters.

    args:
        * model_path - string, filepath of model being evaluated
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling

    Returns an array of labels, and an array of results per template,
    each consists of a tuple containing a pandas dataframes
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    cxr_sex_labels = ['Female', 'Male']

    cxr_sex_templates = [
        # '{}',
        #                      'the patient is a {}',
        "the patient's sex is {}",
    ]

    # run zero shot experiment
    sex_labels_path = '../../data/val_sex_labels.csv'
    results = run_zero_shot(cxr_sex_labels, cxr_sex_templates, model_path, cxr_filepath=cxr_filepath,
                            final_label_path=sex_labels_path, softmax_eval=False, context_length=context_length,
                            pretrained=True, use_bootstrap=True, cutlabels=False)

    results = run_zero_shot(cxr_sex_labels, cxr_sex_templates, model_path, cxr_filepath=cxr_filepath,
                            final_label_path=sex_labels_path, softmax_eval=False, context_length=context_length,
                            pretrained=True, use_bootstrap=True, cutlabels=False)
    pass








