# Denoising color image using autoencoder.

## Description:
    - A autoencoder made of CNN (encoder) and transpose CNN (decoder) to 
    denoise 3-channel image.
    - Dataset used: CelebA.
    - Trained model is saved in .pth file and used to denoise test images.
    
## Output Images:
![Output Image](output_images/output-ayushmann-khurrana-movies-1200x900.jpg)
![Output Image](output_images/output-4cc6f1f365fce288fcae170fa17b6875.jpg)
![Output Image](output_images/output-1955.jpg)
![Output Image](output_images/output-johnny-depp-2016-1200x1256.jpg)
![Output Image](output_images/output-deepika-padukones-straight-from-the-heart-note-about-depression.jpg)
![Output Image](output_images/output-joaquin-phoenix-1004-01-10-2019-11-38-52.jpg)

## Training and Predicting:
    - Use denoise.ipynb to train model.
    - use saved .pth file to get denoised output using main.py file.

