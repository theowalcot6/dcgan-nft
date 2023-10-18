# Dcgan-nft

**Context**
📋 Overview
Generative Adversarial Networks (GANs) are powerful machine learning models capable of generating realistic image, video, and voice outputs. Rooted in game theory, GANs have widespread application: from improving cybersecurity by fighting against adversarial attacks and anonymizing data to preserve privacy to generating state-of-the-art images, colourizing black and white images, increasing image resolution, creating avatars, turning 2D images to 3D, and more.

👩‍💻 Stakeholders
This stakeholder wanted me to build a basic GAN using PyTorch, which then added further convolutional layers to become a more advanced DCGAN that could process the images provided. As this project was more research than outcome-driven, stakeholder management was fairly minimal but I was proactive in my reporting of the project and held regular catch-ups with the project lead to communicate progress

**Challenge**
📈 Data Collection
I built a scraping tool that scrolled through the image feeds on OpenSea and saved them into my storage destination. I was using Google Colab for this project mostly, back when the Colab GPUs were much cheaper, so the scraping script fed into my Google Drive which I then used to store the images.

🗂️ Data Cleaning
The data cleaning process involved ingesting the images into Pytorch tensors. I then had to convert all the images into standard widths, heights and channels to feed into the Discriminator. I also applied advanced data augmentation methods to improve the training by providing a broader training set.

🧨 Modelling
I then developed a Generator, whose job it was to generate an image that the Discriminator couldn't figure out was fake, and a Discriminator, whose job it was to guess that the Generators were fake. The Discriminator loss function was linked to the answers it got wrong. The Generator loss function was linked to the answers the Discriminator got right so that the backpropagation algorithm would force both these models to get better at these jobs.

**Outcome**
🤞 Key Points
The Generator was then evaluated using random Gaussian noise. Then the image was converted back from torch tensors into an actual image (you can see a collection of them in the middle image above). The images were from a broad distribution so without extensive training and the use of parallel GPUs it would be tough to convert onto a solution of HD NFTs. However, as this was more of a research piece, we stopped after seeing some interesting results. As you can see from the images, the Generator started picking up on broad colours and eye shapes, along with a focus on the middle pixels.

🔮 Future Steps
The project was completed and the findings were written up, which was continued by another developer as I rolled off. My advice was to focus on a more constricted distribution if they wanted to go commercial and also to apply a deep learning upscaler to turn the 280 x 280 image into something more HD. I also agreed to work in future with the same stakeholder on the same problem statement but using a diffusion model instead (more on that later...)

All code, as well as the outcome, can be viewed [here](https://colab.research.google.com/drive/1bwqNIUFEKwdj2jH0B1M89M5NkxAVotSY?usp=sharing)
