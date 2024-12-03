import torch
from neuralnet import ESPCN_model, FSRCNN_model, SRCNN_model, VDSR_model, SwinIR
from swinfir.archs.swinfir_arch import SwinFIR
from utils.common import exist_value, to_cpu, ycbcr2rgb, rgb2ycbcr, norm01, denorm01, write_image
import numpy as np
import time
import os

class State:
    def __init__(self, scale, device):
        self.device = device
        self.lr_image = None
        self.sr_image = None
        self.tensor = None
        self.move_range = 3
        self.window_size_swinir = 8
        self.window_size_swinfir = 12
        self.scale = scale

        dev = torch.device(device)
        
        # Next to be replaced with the other model
        #self.SRCNN = SRCNN_model().to(device)
        #model_path = "sr_weight/SRCNN-955.pt"
        #self.SRCNN.load_state_dict(torch.load(model_path, dev, weights_only=True))
        #self.SRCNN.eval()

        self.FSRCNN = FSRCNN_model(scale).to(device)
        model_path = f"sr_weight/x{scale}/FSRCNN-x{scale}.pt"
        self.FSRCNN.load_state_dict(torch.load(model_path, dev, weights_only=True))
        self.FSRCNN.eval()

        self.ESPCN = ESPCN_model(scale).to(device)
        model_path = f"sr_weight/x{scale}/ESPCN-x{scale}.pt"
        self.ESPCN.load_state_dict(torch.load(model_path, dev, weights_only=True))
        self.ESPCN.eval()

        # SwinFIR_SRx2.pth
        model_path = f"sr_weight/x{scale}/SwinFIR_SRx{scale}.pth"
        self.SwinFIR = SwinFIR(
            upscale=scale,
            in_chans=3,
            img_size=60,
            window_size=self.window_size_swinfir,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='SFB').to(device)
        self.SwinFIR.load_state_dict(torch.load(model_path, dev, weights_only=True)["params_ema"], strict=True)
        self.SwinFIR.eval()

        model_path = f"sr_weight/x{scale}/SwinIR-M_x{scale}.pth"
        self.SwinIR = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=self.window_size_swinir,
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
        
        swinfir = self.sr_image.clone()
        espcn = self.sr_image.clone()
        fsrcnn = self.sr_image.clone()
        swinir = self.sr_image.clone()

        neutral = (self.move_range - 1) / 2
        move = act.type(torch.float32)
        move = (move - neutral) / 255
        moved_image = self.sr_image.clone()
        for i in range(0, self.sr_image.shape[1]):
            moved_image[:, i] += move[0]

        self.lr_image = self.lr_image.to(self.device)
        self.sr_image = self.sr_image.to(self.device)

        with torch.no_grad():
            if exist_value(act, 3):
                print("ESPCN")
                espcn = to_cpu(self.ESPCN(self.lr_image))
            if exist_value(act, 4):                
                print("SwinFIR")
                lr_changed = self.lr_image.clone()
                print(lr_changed.size())
                window_size = 12
                # Pad the image to be divisible by the window size
                _, _, h, w = lr_changed.size()
                mod_pad_h = (self.window_size_swinfir - (h % self.window_size_swinfir)) % self.window_size_swinfir
                mod_pad_w = (self.window_size_swinfir - (w % self.window_size_swinfir)) % self.window_size_swinfir

                # Apply reflective padding
                lr_changed = torch.nn.functional.pad(
                    lr_changed,
                    (0, mod_pad_w, 0, mod_pad_h),  # (left, right, top, bottom)
                    mode='replicate'
                )

                print(f"Original: {h}x{w}, Padded: {h + mod_pad_h}x{w + mod_pad_w}")
                # Change from ycbcr to rgb
                lr_changed = denorm01(lr_changed).clone()
                
                # Assuming lr_changed is a batch of images
                images_rgb = torch.stack([ycbcr2rgb(image.clone()) for image in lr_changed]).to(self.device)
                #Save image before
                images_rgb_ = to_cpu(images_rgb.type(torch.uint8))
                write_image("before_test.png", images_rgb_[0])
                
                swinfir = to_cpu(self.SwinFIR(images_rgb))
                # Crop back to the original scale
                h_orig, w_orig = h * self.scale, w * self.scale
                swinfir = swinfir[..., :h_orig, :w_orig]
                swinfir = swinfir.float().cpu().clamp(0, 255)
                swinfir = swinfir.to(torch.uint8)
                
                print(f"Super-resolved: {swinfir.shape[-2:]}, Cropped: {h * self.scale}x{w * self.scale}")
                #Save image after
                write_image("after_test.png", swinfir[0])
                print("Wrote images")
                
                # # Change from rgb to ycbcr
                swinfir = torch.stack([rgb2ycbcr(image.clone()) for image in swinfir.clone()]).to(self.device)
                # # Normalize the image again
                swinfir = to_cpu(norm01(swinfir.clone()))
            if exist_value(act, 5):
                print("FSRCNN")
                fsrcnn = to_cpu(self.FSRCNN(self.lr_image))
            if exist_value(act, 6):
                print("SwinIR")
                lr_changed = self.lr_image.clone()
                # Pad the image to be divisible by the window size
                _, _, h_old, w_old = lr_changed.size()
                h_pad = (h_old // self.window_size_swinir + 1) * self.window_size_swinir - h_old
                w_pad = (w_old // self.window_size_swinir + 1) * self.window_size_swinir - w_old
                lr_changed = torch.cat([lr_changed, torch.flip(lr_changed, [2])], 2)[:, :, :h_old + h_pad, :]
                lr_changed = torch.cat([lr_changed, torch.flip(lr_changed, [3])], 3)[:, :, :, :w_old + w_pad]

                # Change from ycbcr to rgb              
                lr_changed = denorm01(lr_changed)
                lr_changed = lr_changed.type(torch.uint8)
                
                # Assuming lr_changed is a batch of images
                images_rgb = [ycbcr2rgb(image) for image in lr_changed]
                # Stack the converted images back into a single tensor
                images_rgb = torch.stack(images_rgb).to(self.device)
                
                start_time = time.time()
                swinir = to_cpu(self.SwinIR(images_rgb))
                end_time = time.time()
                

                swinir = swinir[..., :h_old * self.scale, :w_old * self.scale]
                
                # Change from rgb to ycbcr
                swinir = [rgb2ycbcr(image) for image in swinir]
                swinir = torch.stack(swinir).to(self.device)
                
                # Normalize the image again
                swinir = to_cpu(norm01(swinir))

        self.lr_image = to_cpu(self.lr_image)
        self.sr_image = moved_image
        act = act.unsqueeze(1)
        act = torch.concat([act, act, act], 1)
        self.sr_image = torch.where(act==3, espcn,  self.sr_image)
        self.sr_image = torch.where(act==4, swinfir,  self.sr_image)
        self.sr_image = torch.where(act==5, fsrcnn,   self.sr_image)
        self.sr_image = torch.where(act==6, swinir, self.sr_image)

        self.tensor[:,0:3,:,:] = self.sr_image
        self.tensor[:,-64:,:,:] = inner_state