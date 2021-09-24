# STGAN
"Neural artistic traslator using novel cGAN architechure."
## About

This repository introduces two cGAN based neural style tranfer (NST) model architechures to overcome two indentified fitfalls in general NST model.
The first and major features introducing here is that a trained model will support several style to traslate with seperately training on each style doamin.
And the model will only extract semantic meaning of the style and the alias features in the generated image will be significatly low.

## Model Discription
in the research we introduce two distinct model architectures with different bias to two fuctionalities.
### Architecture : 1
In the Geneerative model U-net based encoder-decoder model implement but the encdoer takes 6 channels as the input. Output from the generator model will feed into seperate discriminator model to calculate the similarity between generated image - style image and calculate similarity between generated image - content image to evaluate semantic style representation and object rescontruction respectively. Because of the complete GAN architechure supports two inputs, in the trained model you can both style image and content image that want embed style into not like in usual NST model.
![Architechure-1 high-level representation](/logs/appr_1_gan.png)

### Architecture : 2

## Installation
installation procedure
### Testing
testing code
### Training
training codes in different model architechure
## Credit
### Abstract
### Aitation
