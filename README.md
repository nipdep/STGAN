# STGAN
"Neural artistic traslator using novel cGAN architechure."
## About

This repository introduces two cGAN based neural style tranfer (NST) model architechures to overcome two indentified fitfalls in general NST model.
The first and major features introducing here is that a trained model will support several style to traslate with seperately training on each style doamin.
And the model will only extract semantic meaning of the style and the alias features in the generated image will be significatly low.
![Model generated NST images](/logs/appr_1_eval.png)
## Model Discription
In the research we introduce two distinct model architectures with different bias to two fuctionalities.
### Architecture : 1
In the Geneerative model U-net based encoder-decoder model implement but the encdoer takes 6 channels as the input. Output from the generator model will feed into seperate discriminator model to calculate the similarity between generated image - style image and calculate similarity between generated image - content image to evaluate semantic style representation and object rescontruction respectively. Because of the complete GAN architechure supports two inputs, in the trained model you can both style image and content image that want embed style into not like in usual NST model.
![Architechure-1 high-level representation](/logs/appr_1_gan.png)

### Architecture : 2
this architecture is designed in a way such that itself reinforce on the encoding precoss while training to ensure from the model collups or complex estimation function generation. The model have seperate encoder sub-model to style image and content image encoding process, And also those sub-model will resuse as the decorator to generate adverial loss of generated image against both the passing content and style image. Due the natual of GAN training the encoder model will finally trained in a way that make optimal latent vector to build generative image while getting minimal loss agaist both content and style image. This approach almost work like the CycleGAN ensure the optimal estimation generation without doing mush operation like CycleGAN.
![Architechure-1 high-level representation](/logs/appr_2_gan.png)

## Installation
installation procedure
### Testing
testing code
### Training
training codes in different model architechure
## Credit
### Abstract
### Aitation
