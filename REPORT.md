# Final Report
## Advanced Programming, IT-2202
### Dana Uteusheva, Sabina Alzhanova, Olzhas Nurkaidar
---
Link to our YouTube Video: https://youtu.be/2ybqFf7f3d0 
Link to our code on GitHub: https://github.com/utd14/FinalML 
Google Drive Folder: https://drive.google.com/drive/folders/1s_Z9Ddm87QGQhjCpQQ55kKcRhTv1h-DG?usp=drive_link 

## Introduction
Interpreting emotions in art can be challenging due to the diverse range of expression styles, which vary significantly from one artist to another and across different artistic genres. The goal of the project is to create a machine learning model that will be trained to recognize emotions in artwork created in one specific art style. 

The goal of the project is to create a machine learning model that will be trained to recognize emotions in artwork created in one specific art style. 

For our research we used these materials:
- Jesus, I., Cardoso, J., Busson, A. J., Guedes, A. L., Colcher, S., & Milidiu, R. L. (2019). A CNN-based tool to index emotion on anime character stickers. 2019 IEEE International Symposium on Multimedia (ISM). https://doi.org/10.1109/ism46123.2019.00071  (https://www.researchgate.net/publication/338651794_A_CNN-Based_Tool_to_Index_Emotion_on_Anime_Character_Stickers)
- Oyku Eravci. Emotion Detection using CNN. (2022). (https://www.kaggle.com/code/oykuer/emotion-detection-using-cnn/notebook) 
- Skillcate AI. Emotion Detection Model using CNN — a complete guide. (2022). (https://medium.com/@skillcate/emotion-detection-model-using-cnn-a-complete-guide-831db1421fae)
- oarriaga on GitHub. Face classification. (2017). (https://github.com/oarriaga/face_classification) 
- Raut, N. (n.d.). Facial Emotion Recognition Using Machine Learning. (2018). https://doi.org/10.31979/etd.w5fs-s8wd (https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1643&context=etd_projects)
- GeeksforGeeks. Emotion Detection Using Convolutional Neural Networks (CNNs). (https://www.geeksforgeeks.org/emotion-detection-using-convolutional-neural-networks-cnns/)
- https://medium.com/@lfnetclk4/how-to-do-anime-character-head-detection-easily-c03f3292d643
- https://github.com/hysts/anime-face-detector 

The solution aimed to develop a CNN model that will learn to recognize and classify emotions from anime artwork. We collected images to create our own dataset and train the model to identify patterns and features associated with different emotions. Our current project was trained to recognise four different emotions from different artworks in an anime style: Anger, Fear, Sadness and Happiness, take an image as an input, and show what kind of emotion is displayed on it. This way, we developed a model for emotion recognition by training our model and its assessment for the accuracy of recognizing emotions in characters.

## Data and Methods
To collect our data, we used one of the largest collections of anime-styled artworks - [Safebooru](https://safebooru.org/), and [Hydrus Network](https://hydrusnetwork.github.io/hydrus/index.html) for fast downloading from Safebooru. Then, we properly cropped and categorized our images. 

Examples of “**Happy**” (from out dataset):
![](https://lh7-us.googleusercontent.com/2-ARBH2ENtgBv9y3DbiD0v040Aa2toSNJFscpw8QoOdn_RXCingKwOrMLrUFpzlOoDw2CIwRaXMZwbRU317v28kSIeKiCk3jLgZCwIhtc3JKJU2TqxEo19kT33oHSqiakr6dNGKYIhqBLFcMtsKv_m8)

Examples of “**Fear**”:
![](https://lh7-us.googleusercontent.com/RzsVQjrye-WEira2oFel3Yim3fuHiNeI12IjUmkGyNW_kxrWWtrjhX7DIjVUE8X8jExFcPZtRB8fAnIONQ4nt0EYjIwSCq0tMPEJT2eTzPKF-HgXduKWTri_9vDV-pMxiI3wCE5bbTtzQCzD7j3-CXU)

Examples of “**Angry**”:
![](https://lh7-us.googleusercontent.com/17p1afcSE6eQiNEH1TB3ibGsqvadGr8DMWdGZ-0Myv41J7GzGPGssvvbL_-nAGmrQFNqYwKcgEGLnTCgiKF_FtHHyCb5aGSjDd7yAIaY7ydvGO4xhZpSKdN1R5Cl8MmaJtN6tEC5Y5OFEUkHT9BV8uQ)

Examples of “**Sad**”:
![](https://lh7-us.googleusercontent.com/2SGQ_6VvtPf8-HU6A3uM6sS0fzmyPf7Tgh5Bc9MgFJNFOjTj57x7W4T5pJpocLbFoGl34ikNyKvJGvZt2QLWTWlbnNuYvtVNSGRqG4UaU31-07xs6Yf-H2_z6XsH2CoM2hm5aJ5Lgpn1VdfoM3I9H_0)

**Methodology** that we used:
- **numpy** for computing in Python. It was mostly used for numerical operations, also in preparing image data and handling the output of the model predictions.
- **matplotlib** and **pillow (PIL)** for visualization. matplotlib.pyplot is a library in Python for creating static, interactive, and animated visualizations. We used it to create our output with data visualization for training and validation loss, and show the images from the inputs.
- **tensorflow** for building and training the model
- **ImageDataGenerator** for rescaling the pixel values of images for normalization. It’s a class from Keras that allows augmentation and preprocessing of images. This way, it makes the images ready for input into a convolutional neural network.
- **Sequential** layers from Keras that are convenient for creating CNNs: **Conv2D**, **MaxPooling2D**, **Flatten**, **Dense**. 
- Conv2D was used for extracting patterns in images, MaxPooling2D for reducing spatial dimensions of the output volume, Flatten to convert the data to a vector to feed it into Dense layer, Dense layer for classification, a softmax activation function to output probabilities for each class and also image for preprocessing.
- We compiled our model with **adam** and categorical **crossentropy** because we have several classes.
- We used **Google Colab** as our main environment, and connected it to **Google Drive**, in which we uploaded our images for training, validation, and testing.

## Results
![](https://lh7-us.googleusercontent.com/q6GRyzPLmvNU6wAch4H_t_ZcVlBTgcrzDwY8iSlN3p4t-1ypA8m6PKQyMU3ninxbjr4fRVPd5mzdhrbVg4xtOD0eE1kMuMGcDdVunXaLpBhbgneaHoFMV7ONNqCRqyb1xOUMdT9IywsqkkHMHQyqXaY)

![](https://lh7-us.googleusercontent.com/xibLb98v0r9jsLEtW8rwFLn2bsBMoqBg-3DIMiFK2TFEsgzbQpdpK63uPbYYFM2i39GRgL7fEQbz-WRvqlSmrxAKGltSyWwCW7eGw-3Py2iADA53jscQeTc6z8F33aKnBa1UW1EKDeBl4GvOz4t35To)

![](https://lh7-us.googleusercontent.com/i6e93_im6UCYpPij2dkTINAa3iaKc1HbHfNH8S9MDC52qsi2i-0iiT91MGWHGaL4LWphyxZ-yq1T_R3BXq7NfpZTRxWplnsaQ2r40AYuZCG0_5fExl3aiJ6_tf0ef5bJ5TjevgaOLbT_y9QXRdDCl0I)

![](https://lh7-us.googleusercontent.com/PPj9MKuLcqwYnZF4cW5IOZhHmK8dWhD3CPKtOvCC_wRlscgCS0971l-hf7fu5kGWqCG268sDbiFYaGj_4r1npcCxi92r47BzzIREBAYin5Sn5DLENoYdLUKq39fdzkK4TUvCDDqaWy_dJncODQDCADM)

![](https://lh7-us.googleusercontent.com/D6tMcDYxUgAT0OJV6rrBbEMxqmZIbXRxL94muhNFBiDoRYAPzHSjD8p3_vGhz0dTr9dMQt_CTMLkx4KOjRzTVFHVHSG5-1J3w03XT6YO1nUc6nCAinWiFHF8f-RAEb4whIk2Cf8n86cbCw9JtTcC2SY)

![](https://lh7-us.googleusercontent.com/2j_rTJMKFOT6T4PMWaRC6_QD01jo5udOYj4hR9vH60t4QvFtVCZ7YyNipoSeFe1oIc17FHBkrPpSRfx7cXQlqGPXxjEab0lL7pjU-o2APuCaiXBC6awup4AEjyBN0fGOuy_d_GPZJKvMPkgm48YSUhs)

## Discussion
Our results showed that our model was able to successfully recognise patterns in art and more often than not correctly recognise the emotions expressed in it. However, it may exhibit inaccuracies due to insufficient number of images in the dataset. If you take a look, the validation loss diverging from the training loss as epochs increase, or if the validation accuracy plateaus or decreases while training accuracy continues to climb. This suggests overfitting.

The project should also take into the account that the predicted emotion might not always align with the expected outcome based on human judgment. A smile on a character's face doesn't necessarily indicate happiness, plot wise there might actually be hiding a lot of suffering beneath this façade.

Our next steps will involve expanding our database, learning more about Machine Learning, and enhancing the model, working on its complexity. We will also try to account for emotions that are hard to identify, for example - a confusion between a neutral and a confused face, a sarcastic smile, and et cetera. Getting qualitative assessments from human observers will also be useful for improving the accuracy of the model. 

In short, we will:
- fine-tune the model to improve accuracy in emotion recognition
- collect a larger and more diverse dataset to enhance model generalization
- explore more techniques to further develop our model
- account for psychological parts of the project.