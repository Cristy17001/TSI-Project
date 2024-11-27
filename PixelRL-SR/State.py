import torch
from neuralnet import ESPCN_model, FSRCNN_model, SRCNN_model, VDSR_model, SwinIR
from utils.common import exist_value, to_cpu
import numpy as np
import os

class State:
    def __init__(self, scale, device):
        self.device = device
        self.lr_image = None
        self.sr_image = None
        self.tensor = None
        self.move_range = 3
        self.window_size = 8
        self.scale = scale

        dev = torch.device(device)
        
        # Next to be replaced with the other model
        self.SRCNN = SRCNN_model().to(device)
        model_path = "sr_weight/SRCNN-955.pt"
        self.SRCNN.load_state_dict(torch.load(model_path, dev, weights_only=True))
        self.SRCNN.eval()

        self.FSRCNN = FSRCNN_model(scale).to(device)
        model_path = f"sr_weight/x{scale}/FSRCNN-x{scale}.pt"
        self.FSRCNN.load_state_dict(torch.load(model_path, dev))
        self.FSRCNN.eval()

        self.ESPCN = ESPCN_model(scale).to(device)
        model_path = f"sr_weight/x{scale}/ESPCN-x{scale}.pt"
        self.ESPCN.load_state_dict(torch.load(model_path, dev, weights_only=True))
        self.ESPCN.eval()

        #/content/TSI-Project/PixelRL-SR/sr_weight/x2/SwinIR-M-x2.pth
        # BATCH SIZE ESTÁ HARDCODED A 64 TEM QUE SER ASSIM
        model_path = f"sr_weight/x{scale}/SwinIR-M_x{scale}.pth"
        self.SwinIR = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=self.window_size,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
        self.SwinIR.load_state_dict(torch.load(model_path, dev, weights_only=True)['params'], strict=True)
        self.SwinIR.eval()

    def reset(self, lr, bicubic):
        self.lr_image = lr 
        self.sr_image = bicubic
        b, _, h, w = self.sr_image.shape
        previous_state = torch.zeros(size=(b, 64, h, w), dtype=self.lr_image.dtype)
        self.tensor = torch.concat([self.sr_image, previous_state], dim=1)

    def set(self, lr, bicubic):
        self.lr_image = lr
        self.sr_image = bicubic
        self.tensor[:,0:3,:,:] = self.sr_image

    def step(self, act, inner_state):
        act = to_cpu(act)
        inner_state = to_cpu(inner_state)
        
        srcnn = self.sr_image.clone()
        espcn = self.sr_image.clone()
        fsrcnn = self.sr_image.clone()
        swinir = self.sr_image.clone()

        neutral = (self.move_range - 1) / 2
        move = act.type(torch.float32)
        move = (move - neutral) / 255
        moved_image = self.sr_image.clone()
        for i in range(0, self.sr_image.shape[1]):
            moved_image[:,i] += move[0]

        self.lr_image = self.lr_image.to(self.device)
        self.sr_image = self.sr_image.to(self.device)

        with torch.no_grad():
            if exist_value(act, 3):
                espcn = to_cpu(self.ESPCN(self.lr_image))
            if exist_value(act, 4):
                srcnn[:, :, 8:-8, 8:-8] = to_cpu(self.SRCNN(self.sr_image))
            if exist_value(act, 5):
                fsrcnn = to_cpu(self.FSRCNN(self.lr_image))
            if exist_value(act, 6):
              # pad input image to be a multiple of window_size
              _, _, h_old, w_old = self.lr_image.size()
              h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
              w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
              # Pad the vertical and horizontal sides of the image with the flip of the image just to be able to process it
              lr_changed = torch.cat([self.lr_image, torch.flip(self.lr_image, [2])[:, :, :h_pad, :]], 2)
              lr_changed = torch.cat([lr_changed, torch.flip(lr_changed, [3])[:, :, :, :w_pad]], 3)
              print(lr_changed)
              swinir = to_cpu(self.SwinIR(lr_changed))
              swinir = swinir[..., :h_old * self.scale, :w_old * self.scale]

        self.lr_image = to_cpu(self.lr_image)
        self.sr_image = moved_image
        act = act.unsqueeze(1)
        act = torch.concat([act, act, act], 1)
        self.sr_image = torch.where(act==3, espcn,  self.sr_image)
        self.sr_image = torch.where(act==4, srcnn,  self.sr_image)
        self.sr_image = torch.where(act==5, fsrcnn,   self.sr_image)
        self.sr_image = torch.where(act==6, swinir, self.sr_image)

        self.tensor[:,0:3,:,:] = self.sr_image
        self.tensor[:,-64:,:,:] = inner_state
