import torch as th
import os
import pickle
import json
import cv2
import numpy as np
import torch 
import clip
from PIL import Image
from torch.nn.functional import normalize
import sys
sys.path.append("..") 
from home import get_project_base

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
device = "cuda:2"

model, preprocess = clip.load("ViT-B/32", device=device)

# print(get_project_base())

def get_video_input_gene(video_path, video_name):
    """
    Downsample video frames for Generic video datasets(TVSum, SumMe, Youtube, OVP). 
    """
    vidcap = cv2.VideoCapture(video_path)
    height = cv2.VideoCapture(video_path).get(4)
    width = cv2.VideoCapture(video_path).get(3)
    img = [] # the set of frames of the given video
    success, image = vidcap.read() # (360, 540, 3)
    while success:
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = preprocess(image)
        img.append(image.cpu().numpy())
        success, image = vidcap.read()

    video_length = len(img)
    print('video length: {}'.format(video_length))

    # tvsum 30fps, sample the video frame every 15 frames
    # summe 25fps, sample the video frame every 15 frames
    ret_img = []
    for i in range(0, video_length, 15):
        ret_img.append(img[i])
    
    if len(ret_img)%2==1:
        ret_img.pop()

    ret_img = np.array(ret_img) # T *3 *H * W 

    return torch.from_numpy(ret_img).float().to(device) 


def get_video_input_wiki(video_path, video_name):
    """
    Downsample video frames for Instructional video datasets(Wikihow). 
    """
    vidcap = cv2.VideoCapture(video_path)
    height = cv2.VideoCapture(video_path).get(4)
    width = cv2.VideoCapture(video_path).get(3)
    img = [] # the set of frames of the given video
    success, image = vidcap.read() # (360, 540, 3)
    while success:
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = preprocess(image)
        img.append(image.cpu().numpy())
        success, image = vidcap.read()

    video_length = len(img)
    print('video length: {}'.format(video_length))
    ret_img = []
    for i in range(0, video_length, 8):
        ret_img.append(img[i])
    
    ret_img = np.array(ret_img)

    return torch.from_numpy(ret_img).float().to(device) 


def extract_tvsum_features():
    BASE = get_project_base()
    videos_path = os.path.join(BASE, 'raw_data', 'TVSum', 'mp4video')
    videos = os.listdir(videos_path)

    frame_feat = dict()
    for video in videos:
        video_path = os.path.join(videos_path, video)
        video_name, _ = video.split('.') # xxx.mp4
        video_ = get_video_input_gene(video_path, video_name)
        print("size: {}".format(video_.shape))
        
        with torch.no_grad():
            image_features = model.encode_image(video_)

        image_features = normalize(image_features, p=2.0, dim = 1)

        frame_feat[video_name] = image_features.detach().cpu().numpy()
        print('video {} complete.'.format(video))
    
    save_path = os.path.join(BASE, 'dataset', 'v_feat', 'tvsum_clip_feats_norm.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(frame_feat, f)
        print('feats save successfully.')


def extract_tvsum_text_features():
    BASE = get_project_base()
    text_path = os.path.join(BASE, 'raw_data', 'TVSum', 'generated_texts', 'shot20_new')
    videos = os.listdir(text_path)
    text_feat = dict()
    for video in videos:
        video_name = video[:-5] # xxx.mp4
        text_file = video_name + '.json'
        path_text =os.path.join(text_path, text_file)
        texts = []
        with open(path_text, 'r') as f:
            load_dict = json.load(f)
            for idx in range(len(load_dict)):
                texts.append(load_dict[idx]['sentence'])

        print("size: {}".format(len(texts)))
        text = clip.tokenize(texts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text)

        text_features = normalize(text_features, p=2.0, dim = 1)

        text_feat[video_name] = text_features.detach().cpu().numpy()
        print('video {} complete.'.format(video))
    
    save_path = os.path.join(BASE, 'dataset', 'w_feat', 'tvsum_clip_shot20_new_text_feats_norm.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(text_feat, f)
        print('feats save successfully.')


def extract_wikihow_features():
    BASE = get_project_base()
    videos_path = os.path.join(BASE, 'raw_data', 'WikiHow', 'mp4video')
    videos = os.listdir(videos_path)
    frame_feat = dict()

    for video in videos:
        video_path = os.path.join(videos_path, video)
        video_name = video[:-4] # note here use [:-4], because '.' exist in some videos' filename. 
        video_ = get_video_input_wiki(video_path, video_name)
        print("size: {}".format(video_.shape))
        
        with torch.no_grad():
            image_features = model.encode_image(video_)
        
        image_features = normalize(image_features, p=2.0, dim = 1)

        frame_feat[video_name] = image_features.detach().cpu().numpy()
        print('video {} complete.'.format(video))
    
    save_path = os.path.join(BASE, 'dataset', 'v_feat', 'wikihow_clip_feats_8_norm.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(frame_feat, f)
        print('feats save successfully.')
