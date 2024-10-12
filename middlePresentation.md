# Presentation Structure

## Introduction

The paper focuses on leveraging advanced computer techniques to enhance the quality of images. It introduces the concept of Deep Learning, specifically utilizing Generative Adversarial Networks (GANs), to achieve Single Image Super-Resolution (SISR). This technique effectively transforms low-quality images into high-quality ones.

Furthermore, the research highlights the broader implications of image quality improvement, such as aiding object identification in images. To address this, the paper proposes a novel approach employing Reinforcement Learning. Multiple agents are involved, each selecting a method to enhance image quality. The image is then updated based on the chosen method. The arrangement of agents encourages them to learn and select the most effective method for improving image quality.

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
- **ESPCN (Efficient Sub-Pixel CNN)**: It is a model that reconstructs a high-resolution version of an image given a low-resolution version. It leverages efficient "sub-pixel convolution" layers, which learns an array of image upscaling filters.
- **FSRCNN (Fast Super-Resolution Convolutional Neural Network)**: This model is designed for fast super-resolution, using a smaller network with fewer parameters to achieve better performance and faster inference times.
- **SRCNN (Super-Resolution Convolutional Neural Network)**: This is a deep learning approach to image super-resolution. It is a straightforward CNN that learns to map low-resolution images to high-resolution ones.
- **VDSR (Very Deep Super Resolution)**: This model uses a deeper architecture(it has 20 weight layers which is much deeper compared with SRCNN which only got 3 layers), leveraging residual learning to improve training and performance on super-resolution tasks.

  
- Explain the code a little more (used models and so on)
- Show images describing the input and output of the code.

## Work Plan

- Present the work plan for the project.
- Mention the need to execute the code and understand its functioning.
- Understand the underlying models and evaluate their effectiveness (mention how it is being measured).

## Conclusion

Summarize the key points discussed in the presentation and highlight the importance of multi-agent artificial intelligence in the given context.
