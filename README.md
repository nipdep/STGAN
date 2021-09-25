# STGAN
"Neural artistic traslator using novel cGAN architechure."
## About

This repository introduces two cGAN based neural style tranfer (NST) model architechures to overcome two indentified fitfalls in general NST model.
The first and major features introducing here is that a trained model will support several style to traslate with seperately training on each style doamin.
And the model will only extract semantic meaning of the style and the alias features in the generated image will be significatly low. \
<img src="/logs/appr_1_eval.png" alt="Model generated NST images" style="width:500px;text-align:center"/>
## Model Discription
In the research we introduce two distinct model architectures with different bias to two fuctionalities.
### Architecture : 1
In the Geneerative model U-net based encoder-decoder model implement but the encdoer takes 6 channels as the input. Output from the generator model will feed into seperate discriminator model to calculate the similarity between generated image - style image and calculate similarity between generated image - content image to evaluate semantic style representation and object rescontruction respectively. Because of the complete GAN architechure supports two inputs, in the trained model you can both style image and content image that want embed style into not like in usual NST model. \
<img src="/logs/appr_1_gan.png" alt="Architechure-1 high-level representation" style="width:500px;text-align:center"/>

### Architecture : 2
this architecture is designed in a way such that itself reinforce on the encoding precoss while training to ensure from the model collups or complex estimation function generation. The model have seperate encoder sub-model to style image and content image encoding process, And also those sub-model will resuse as the decorator to generate adverial loss of generated image against both the passing content and style image. Due the natual of GAN training the encoder model will finally trained in a way that make optimal latent vector to build generative image while getting minimal loss agaist both content and style image. This approach almost work like the CycleGAN ensure the optimal estimation generation without doing mush operation like CycleGAN. \
<img src="/logs/appr_2_gan.png" alt="Architechure-1 high-level representation" style="width:500px;text-align:center"/>

## Installation
installation procedure
### Testing
testing code
### Training
training codes in different model architechure
## Credit
### Abstract
A neural artistic style transformation (NST) model can modify the appearance of a simple image by adding the style of a famous image. Even though the transformed images do not look exactly like artworks by the same artist of the respective style images, the generated images are appealing. Generally, a trained NST model is specialized for a style and that style is represented by a single image. In the first approach we introduce a conditional adversarial network for neural style transfer. The proposing supports more than one style to translate and embeds fewer alias artifacts on the generated image. In the second approach, we improve the first approach to obtain a model which understands the related style class of the input style image and impose the general features of the identified class into the content image.The experiments demonstrate that the resulting images are semantically highly accurate according to both content and style domains.
### Citation
```
{didn't publish yet.}
```
