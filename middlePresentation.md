# Presentation Structure

## Introduction

The paper focuses on leveraging advanced computer techniques to enhance the quality of images. It introduces the concept of Deep Learning, specifically utilizing Generative Adversarial Networks (GANs), to achieve Single Image Super-Resolution (SISR). This technique effectively transforms low-quality images into high-quality ones.

Furthermore, the research highlights the broader implications of image quality improvement, such as aiding object identification in images. To address this, the paper proposes a novel approach employing Reinforcement Learning. Multiple agents are involved, each selecting a method to enhance image quality. The image is then updated based on the chosen method. The arrangement of agents encourages them to learn and select the most effective method for improving image quality.

(A minha correção:)

Recent advancements in deep learning have significantly impacted not only image classification but also various image processing tasks such as filtering, colorization, generation, and translation. These tasks utilize neural networks, which can be categorized into two primary structures: Convolutional Neural Networks (CNNs) for image recognition and Fully Convolutional Networks (FCNs) for tasks requiring pixel-level precision.

Building upon the success of Deep Q-Networks (DQNs), which demonstrated human-level performance in playing Atari games, there has been growing interest in applying Deep Reinforcement Learning (RL) to image processing. However, earlier methods in deep RL were limited to global actions (e.g., image cropping or color enhancement), making them unsuitable for pixel-wise manipulations required in more complex tasks such as image denoising.

The text proposes a novel approach called PixelRL, a multi-agent reinforcement learning (RL) framework designed for image processing tasks at the pixel level. In pixelRL, each pixel is controlled by an agent that learns optimal actions to maximize rewards across all pixels, allowing for tasks like local color enhancement, and saliency-driven image editing. To overcome the computational challenge of handling millions of agents (one per pixel), the method leverages fully convolutional networks (FCNs), enabling efficient parameter sharing among agents.

Additionally, the paper introduces reward map convolution, a technique that allows agents to consider both their own pixel’s future state and neighboring pixels, enhancing learning efficiency. This approach is unique in its interpretability, as it makes it possible to observe the agents’ actions. The experimental results show that pixelRL achieves performance comparable to or better than traditional CNN-based methods on various image processing tasks.
## Applications

The use of multi-agent artificial intelligence for improving image quality has various applications:

- **Medical Imaging:** High-resolution images can aid doctors and medical professionals in making more accurate diagnoses.
- **Surveillance and Security:** Enhanced image quality can facilitate the identification of individuals or objects with greater precision.
- **Photography and Art:** Artists and photographers can utilize image enhancement techniques to restore old or low-quality images, preserving valuable visual content.
- **Forensics:** High-resolution images can offer more detailed information for forensic investigations, aiding in evidence analysis.

These are just a few examples of the wide range of applications for improving low-resolution images using artificial intelligence. The field of image enhancement has numerous other important applications across various industries and domains.

## Code Execution Challenges and What the code does

During the execution of the code from [PixelRL-SR](https://github.com/Nhat-Thanh/PixelRL-SR) mentioned in the paper, we encountered technical difficulties related to an outdated library, specifically PyTorch. When attempting to run the code, we encountered an error indicating a vulnerability in the load function, which could potentially allow for the execution of code from a pickle-formatted file.

To address this vulnerability, we applied a solution by adding the directive "weights_only=True" to each instance of the torch.load function in the code. This modification ensured that only the weights of the model were loaded, mitigating the potential security risk associated with executing arbitrary code.

After implementing this solution, the code executed successfully without any further issues.

So, after executing successfully the code, our goal was to explore the used models and libraries in this project. We identified many different libraries such as:
- **PyTorch**: A popular deep learning framework that provides dynamic computation graphs, making it flexible and user-friendly for research and prototyping. The libraries torchvision and torchmetrics are as well often used for image processing tasks.
- **TensorFlow/Keras**: Another widely-used deep learning framework. Keras provides a high-level interface to build and train models easily, while TensorFlow provides a more extensive set of tools and libraries for building complex models.
- **OpenCV**: A powerful library for computer vision tasks, often used for image processing, manipulation, and augmentation. It can complement the functionality of PyTorch and other frameworks.
- **NumPy**: While not specific to deep learning, NumPy is essential for numerical computations and handling image data, especially when working with arrays and matrices.
- **scikit-image**: A collection of algorithms for image processing in Python. It can be useful for tasks like filtering, transforming, and analyzing images.
- **Matplotlib**: Often used for visualization of images and results, helping in debugging and presenting findings.

Besides that, we found some models (neural networks) that are being used, such as:
- **ESPCN (Efficient Sub-Pixel CNN)**:  Short version: It is a model that reconstructs a high-resolution version of an image given a low-resolution version. In ESPCN, the network is a combination of several convolutional layers and a sub-pixel convolutional layer, the LR image is upscaled at the last pixel shuffle stage. Thus, a big advantage of ESPCN is that it has a higher computation speed. 
Long version ( if professor asks about it): ESPCN has been proposed to add an efficient sub-pixel convolutional layer to the CNN network. ESPCN increases the resolution at the very end of the network. In ESPCN, the upscaling step is handled by the last layer, which means the smaller size LR image is directly fed to the network. Thus, there is no need to use the interpolation method. The network is capable of learning a better LR to HR mapping compared to an interpolation filter upscaling before feeding into the network. Due to the reduced input image size, a smaller filter size can be used to extract features. The computational complexity and memory cost is reduced so that the efficiency can be greatly enhanced. ** not needed (This is why ESPCN become an ideal choice for the super-resolution of HD videos in real-time)  **
- **FSRCNN (Fast Super-Resolution Convolutional Neural Network)**: This model is designed for fast super-resolution, using a smaller network with fewer parameters to achieve better performance and faster inference times.
- **SRCNN (Super-Resolution Convolutional Neural Network)**: This is a deep learning approach to image super-resolution. It is a straightforward CNN that learns to map low-resolution images to high-resolution ones.
- **VDSR (Very Deep Super Resolution)**: This model uses a deeper architecture(it has 20 weight layers which is much deeper compared with SRCNN which only got 3 layers), leveraging residual learning to improve training and performance on super-resolution tasks.

- Explain the code a little more (used models and so on)
- The code is equipped with a dataset that provides a total of 391 images to train each of the models implemented. Firstly, it is used a  set of 91 images of basic, low-detailed nature-based images, followed by a mix of 100 random-themed images with varying complexities and textures, and then, the remaining 200 are royalty-free images from the BSD (Berkeley Segmentation Dataset) testing set, which is commonly used in computer vision tasks. It is important to note that the BSD is frequently used for evaluating natural edge detection that includes not only object contours but also object interior boundaries and background boundaries.
Meanwhile a set of 5 images is used to test models with scale factors of x2, x3, and x4, where the result is the average Peak signal-to-noise ratio, that is , the quantity measured as the ratio between the power of the signal noise and a signal’s maximum power, of all images.
- At the end of training, an snapshot is taken to save the training process and track the models improvements.
- To validate the model effectively, that is , to assure that the model itself is learning effectively, the model performance is checked with a set of 14 images , with nature-based themes, while each model is being trained. This validation step ensures that the model generalizes well beyond the training data and continues to improve in real-world scenarios.
- Finally, to better understand the model's behavior and to track the learning process, a sequence of action maps are drew throughout the model evolution. Thus, if the model is not performing well on certain images or regions, the action map can pinpoint where and why the model is failing, guiding improvements to the model's architecture or training process.


- Show images describing the input and output of the code.

![Butterfly.png in Set5, Bicubic (left), PixelRL-SR x2 (center), High Resolution (right).](PixelRL-SR\example.png "Before")

*Butterfly.png in Set5, Bicubic (left), PixelRL-SR x2 (center), High Resolution (right).*

- For each scale factor, to each used model is evaluated a classification score, that ranges between 20.00-35.00.
![Butterfly.png in Set5, Bicubic (left), PixelRL-SR x2 (center), High Resolution (right).](PixelRL-SR\scores.png "Score Table")
*Classification scores for each model evaluated, given each of the separate subsets of the actual dataset + Urban 100, being the Bold texts the best resuls*

## Work Plan

- Present the work plan for the project.

- Given that the code implementation was not documented, first of all, it is necessary to follow and understand precisely the actual structure of the code, as well as their inner mechanisms, classes and functions.
- On the one hand , we acknoledged the importance of checking and verifying the models already implemented, in order to check possible errors that are live , and improve their performance. For that instance, we need also to understand these underlying models and verify how their effectiveness is being measured and evaluated.
- On the other hand, we aim forward to implement new, and more recent , training models, to improve the super-resolutioning process itself, and seek more efficiency and performance, since the actual models still struggle a lot when using panoramic images, that , in the sense of hence, tend to have high-resolutions while being heavily detailed and dense. Thus, one of the crucial parts of the work plan is that all the members are commited to search for these new models to be implemented, as well as for trying to implement them, in a way to find new PSNR scores and, subsequently, compare these models newly introduced. 

## Conclusion

- Finally , we are all set to reproduce the work carried out, as well as to continue this work with the most recent models and implementations available.
