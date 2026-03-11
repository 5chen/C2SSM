import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.c2ssm_arch import C2SSM

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

from comput_psnr_ssim import calculate_ssim as ssim_gray
from comput_psnr_ssim import calculate_psnr as psnr_gray


def check_image_size(x, window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))


def main():
    parser = argparse.ArgumentParser(description='C2SSM Inference')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image or folder')
    parser.add_argument('-g', '--gt', type=str, required=True,
                        help='Ground truth image folder')
    parser.add_argument('-w', '--weight', type=str, required=True,
                        help='Path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results/C2SSM',
                        help='Output folder')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--save_img', action='store_true',
                        help='Save output images')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = C2SSM().to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device)['params'], strict=False)
    model.eval()
    print_network(model)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    ssim_all = 0
    psnr_all = 0
    lpips_all = 0
    num_img = 0
    pbar = tqdm(total=len(paths), unit='image')

    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        file_name = os.path.basename(path)
        gt_img = cv2.imread(os.path.join(args.gt, file_name), cv2.IMREAD_UNCHANGED)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)
        b, c, h, w = img_tensor.size()
        img_tensor = check_image_size(img_tensor)

        with torch.no_grad():
            output = model.forward(img_tensor)

        output = output[:, :, :h, :w]
        output_img = tensor2img(output)

        ssim = ssim_gray(output_img, gt_img)
        psnr = psnr_gray(output_img, gt_img)
        lpips_value = lpips(
            2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
            2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1
        )
        ssim_all += ssim
        psnr_all += psnr
        lpips_all += lpips_value
        num_img += 1

        if args.save_img:
            save_path = os.path.join(args.output, img_name)
            imwrite(output_img, save_path)

        pbar.update(1)

    pbar.close()
    print('avg_ssim:%f' % (ssim_all / num_img))
    print('avg_psnr:%f' % (psnr_all / num_img))
    print('avg_lpips:%f' % (lpips_all / num_img))


if __name__ == '__main__':
    main()
