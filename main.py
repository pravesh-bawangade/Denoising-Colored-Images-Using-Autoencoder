"""
Author: Pravesh Bawangade
File Name: main.py

"""

import torch
from torchvision import transforms
from torch.autograd import Variable
import PIL.Image as Image
import cv2
import numpy as np
import model as m
import os


def predict_image(image, model):
    """
    Predict function provide predicted output of noisy image.
    :param image: Original Image
    :param model: Model to use
    :return: noisy_imgs , output
    """
    noise_factor = 0.3
    image = Image.fromarray(image.astype('uint8'), 'RGB')

    test_transforms = transforms.Compose(
        [transforms.Resize(128),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = Variable(image_tensor)

    noisy_imgs = image_tensor + noise_factor * torch.randn(*image_tensor.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    noisy_imgs = noisy_imgs.to('cpu')

    output = model(noisy_imgs)
    output = output.detach().numpy()
    return noisy_imgs, output


def main():
    """
    Main function to load data from file and store predicted output in given path.
    Also display images on screen.
    :return: None
    """
    model = m.ConvDenoiser()
    model.eval()
    model.load_state_dict(torch.load('trained_models/train-epoch9.pth', map_location=torch.device('cpu')))

    root = 'images/'
    img_list = os.listdir(root)

    for i in range(len(img_list)):
        print(root + img_list[i])
        if img_list[i] != ".DS_Store":
            image = cv2.imread(root + img_list[i])
            image = cv2.resize(image,(130,130))

            noisy_imgs, output = predict_image(image, model)

            noisy_imgs = noisy_imgs[0].numpy().transpose(1, 2, 0)
            output = output[0].transpose(1, 2, 0)
            vis = np.concatenate((noisy_imgs, output), axis=1)

            frame_normed = 255 * (vis - vis.min()) / (vis.max() - vis.min())
            frame_normed = np.array(frame_normed, np.int)

            cv2.imwrite("output_images/output-" + img_list[i], frame_normed)
            cv2.imshow("output", vis)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()