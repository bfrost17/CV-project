import os
import cv2
import pdb
import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from transformers import logging
logging.set_verbosity_error()

class FeatureExtractor():
    def __init__(self, args):
        self.device = args.feature_extractor_device
        self.beta = args.beta
        self.processor = CLIPProcessor.from_pretrained(args.feature_extractor)
        self.model = CLIPVisionModelWithProjection.from_pretrained(args.feature_extractor).to(self.device)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(args.feature_extractor).to(self.device)
        self.text_model.eval()
        self.data_dir = args.data_dir
        self.tmp_dir = args.tmp_dir
        self.last_clip_features = None
        self.last_frame_indices = []


    def __call__(self, video_path, video_id):        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps
        sample_rate = int(fps) * self.beta

        save_path = os.path.join(self.tmp_dir, video_id + '.npz')
        if os.path.exists(save_path):
            data = np.load(save_path)
            clip_features = data['features']
            self.last_clip_features = clip_features
            if 'frame_indices' in data:
                self.last_frame_indices = data['frame_indices'].tolist()
            else:
                self.last_frame_indices = list(range(1, len(clip_features) + 1))
            return clip_features, video_length

        clip_features = []
        frame_indices = []
        print("Extract the clip feature.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) % sample_rate == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = self.processor(images=image, return_tensors="pt").pixel_values
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    feat = self.model(inputs)['image_embeds']
                    clip_features.append(feat.cpu().numpy())
                frame_indices.append(len(frame_indices) + 1)
        print("Finished.")
        cap.release()

        clip_features = np.concatenate(clip_features, axis=0)
        np.savez_compressed(
            save_path,
            features=clip_features,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
        )
        self.last_clip_features = clip_features
        self.last_frame_indices = frame_indices
        
        return clip_features, video_length

    def rank_frames(self, question):
        if self.last_clip_features is None:
            raise RuntimeError("Video frame features have not been extracted.")

        text_inputs = self.processor(
            text=[question],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            text_embeds = self.text_model(**text_inputs).text_embeds
            text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

            frame_embeds = torch.from_numpy(self.last_clip_features).to(
                self.device,
                dtype=text_embeds.dtype,
            )
            frame_embeds = torch.nn.functional.normalize(frame_embeds, dim=-1)
            scores = (frame_embeds @ text_embeds.T).squeeze(1).cpu().numpy()

        return {
            str(frame_index): float(score)
            for frame_index, score in zip(self.last_frame_indices, scores)
        }
