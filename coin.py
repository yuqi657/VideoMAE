import os
import numpy as np
import torch
from torchvision import transforms
import json
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


class COINVideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                crop_size=224, short_side_size=256, new_height=256,
                new_width=340, keep_aspect_ratio=True, num_segment=1,
                num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        videos_dict = {}
        for video_id in os.listdir(self.data_path):
            videos_dict[video_id] = video_id
        """
        item in train/test_list is 
        (video_id, clip_start, clip_end, annotation)
        annotation is (id, segment[st, ed], label)
        """
        data = []
        with open(self.anno_path) as fp:
            coin_data = json.load(fp)
        for item in coin_data:
            if item["video_id"] in videos_dict:
                data.append(item)
        self.data = data

        self.transform = transforms.Compose([
            transforms.Resize(self.crop_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        video_data = self.data[index]
        video_id = video_data["video_id"]
        clip_start = video_data["clip_start"]
        clip_end = video_data["clip_end"]
        annotation = video_data["annotation"]
        video = self.__get_rawvideo(video_id, clip_start, clip_end) # (channel, video_length, height, weight)

        video_target = np.zeros(video.shape[1]//2)
        for anno in annotation:
            start, end = anno["segment"]
            video_target[start - clip_start: end + 1 - clip_start] = int(anno["id"])
        return video, video_target, video_id, {}

    def __len__(self):
        return len(self.data)

    def __get_rawvideo(self, video_id, clip_start, clip_end):
        video_path = os.path.join(self.data_path, video_id)
        try:
            video_frms = os.listdir(video_path)
            video_frms.sort()
            video_frms = video_frms[2*clip_start: 2*(clip_end+1)]

            if len(video_frms) != 64:  # if odd, append last frame to even. cause videomae tublet size is 2*x*x
                print("dont have 64 frames: ", video_id, clip_start, " - ", clip_end, video_frms)
                video_frms.append(video_frms[-1])

            # pre-process frames
            images = []
            for cnt, frame in enumerate(video_frms):
                image_path = os.path.join(video_path, frame)
                images.append(self.transform(Image.open(image_path).convert("RGB"))) # (num_frm, channel, height, weight)

            # convert into tensor
            if len(images) > 0:
                if len(images) == 64:
                    raw_sample_frms = torch.tensor(np.stack(images)) # (video_length, channel, height, weight)
                else:
                    raw_sample_frms = torch.zeros((64, 3, 224, 224))
                    print("dont have 32 s: ", video_id, clip_start, " - ", clip_end)
            else:
                raw_sample_frms = torch.zeros((64, 3, 224, 224))
                print("dont have frames: ", video_id, clip_start, " - ", clip_end)

        except Exception as e:
            print('Exception: ', e)
            raw_sample_frms = torch.zeros((64, 3, 224, 224))

        raw_sample_frms = raw_sample_frms.permute(1, 0, 2, 3) # (channel, video_length, height, weight)
        return raw_sample_frms

