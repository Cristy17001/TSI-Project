import sys
sys.dont_write_bytecode = True

import argparse
from neuralnet import PixelRL_model 
from State import State
import random
import torch
from utils.common import *

# Set random seeds for reproducibility
def set_random_seeds(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Numpy module.
    random.seed(seed_value)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seeds()

# =====================================================================================
# arguments parser
# =====================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--scale",     type=int, default=2,  help='-')
parser.add_argument("--ckpt-path", type=str, default="", help='-')
parser.add_argument("--test-set", type=str, default="SET5", help='-')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

SCALE = FLAG.scale
if SCALE not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3 or 4")

MODEL_PATH = FLAG.ckpt_path
if (MODEL_PATH == "") or (MODEL_PATH == "default"):
    MODEL_PATH = f"checkpoint/x{SCALE}/PixelRL_SR-x{SCALE}.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ACTIONS = 7
GAMMA = 0.95
T_MAX = 5
SIGMA = 0.3 if SCALE == 2 else 0.2

dataset = FLAG.test_set
print(f"Testing on {dataset} dataset")
if dataset == "SET5":
    LS_HR_PATHS = sorted_list(f"dataset/test/x{SCALE}/labels")
    LS_LR_PATHS = sorted_list(f"dataset/test/x{SCALE}/data")
elif dataset == "SET14":
    LS_HR_PATHS = sorted_list(f"dataset/test/Set14_test/x{SCALE}/labels")
    LS_LR_PATHS = sorted_list(f"dataset/test/Set14_test/x{SCALE}/data")
elif dataset == "BSD100":
    LS_HR_PATHS = sorted_list(f"dataset/test/BSD100_test/x{SCALE}/labels")
    LS_LR_PATHS = sorted_list(f"dataset/test/BSD100_test/x{SCALE}/data")
elif dataset == "URBAN100":
    LS_HR_PATHS = sorted_list(f"dataset/test/Urban100_test/x{SCALE}/labels")
    LS_LR_PATHS = sorted_list(f"dataset/test/Urban100_test/x{SCALE}/data")
else:
    raise ValueError("dataset must be SET5, SET14, URBAN100, BSD100")

# =====================================================================================
# Test each image
# =====================================================================================

def main():
    CURRENT_STATE = State(SCALE, DEVICE)

    MODEL = PixelRL_model(N_ACTIONS).to(DEVICE)
    if exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, torch.device(DEVICE), weights_only=True))
    MODEL.eval()

    reward_array = []
    metric_array = []
    for i in range(0, len(LS_HR_PATHS)):
        hr_image_path = LS_HR_PATHS[i]
        lr_image_path = LS_LR_PATHS[i]
        hr = read_image(hr_image_path)
        lr = read_image(lr_image_path)
        lr = gaussian_blur(lr, sigma=SIGMA)
        bicubic = upscale(lr, SCALE)

        bicubic = rgb2ycbcr(bicubic)
        lr = rgb2ycbcr(lr)
        hr= rgb2ycbcr(hr)


        bicubic = norm01(bicubic).unsqueeze(0)
        lr = norm01(lr).unsqueeze(0)
        hr = norm01(hr).unsqueeze(0)

        with torch.no_grad():
            CURRENT_STATE.reset(lr, bicubic)
            sum_reward = 0
            for t in range(0, T_MAX):
                prev_img = CURRENT_STATE.sr_image.clone()
                statevar = CURRENT_STATE.tensor.to(DEVICE)
                actions, _, inner_state = MODEL.choose_best_actions(statevar)

                CURRENT_STATE.step(actions, inner_state)
                # Calculate reward on Y chanel only
                reward = torch.square(hr[:,0:1] - prev_img[:,0:1]) - \
                         torch.square(hr[:,0:1] - CURRENT_STATE.sr_image[:,0:1])

                sum_reward += torch.mean(reward * 255) * (GAMMA ** t)

            sr = torch.clip(CURRENT_STATE.sr_image, 0.0, 1.0)
            psnr = PSNR(hr, sr)
            metric_array.append(psnr)
            reward_array.append(sum_reward)

    print(f"Average reward: {torch.mean(torch.tensor(reward_array) * 255):.4f}",
          f"- PSNR: {torch.mean(torch.tensor(metric_array)):.4f}")

if __name__ == '__main__':
    main()
