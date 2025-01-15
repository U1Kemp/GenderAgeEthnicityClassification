
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the saved model
model_path = 'multitask_mobilenet.pth' # to be updated
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detect_face(image):
    """
    Detects a face in a given PIL image and returns the face as a PIL image cropped with a square boundary.
    """

    # Step 1: Load the cascade
    # The cascade is a pre-trained model that contains the patterns for detecting faces.
    # It is a XML file that contains the features of the face, such as the eyes, nose, mouth, etc.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Step 2: Convert the PIL image to an OpenCV image
    # OpenCV uses BGR color order, whereas PIL uses RGB color order.
    # Therefore, we need to convert the PIL image to BGR color order.
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Step 3: Detect faces in the image
    # The detectMultiScale function takes in the image, the scale factor and the min neighbors as arguments.
    # The scale factor determines how much the image is scaled down at each iteration.
    # The min neighbors determines how many neighbors a face must have to be considered a valid face.
    faces = face_cascade.detectMultiScale(img, 1.05, 7, flags = cv2.CASCADE_DO_CANNY_PRUNING) 
        
    # Step 4: If faces are detected, find the largest face and crop it
    if len(faces) > 0:
        # Sort faces by area (width * height) in descending order and select the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # if face occupies more than 75% of the image width or more than 50% of the image height,
        # then return whole image
        if w >= 0.9 * img.shape[1] or h >= 0.9 * img.shape[0]:
            return image

        # Make the boundary square
        side = max(w, h)*0.99
        x1 = x + (w - side) // 2
        y1 = y + (h - side) // 2
        face = image.crop((x1, y1, x1 + side, y1 + side))

        return face

    else:
        # If no face is detected, return None
        return None


# class DepthwiseSeparableConv(nn.Module):
#     """
#     A class implementing a depthwise separable convolutional layer.
    
#     Depthwise separable convolution is a technique for reducing the number of parameters and computation required for a convolutional layer.
#     It first applies a depthwise convolution (a convolution with a filter that is applied channel-wise) and then a pointwise convolution (a 1x1 convolution).
#     """
#     def __init__(self, in_channels, out_channels, stride=1):
#         """
#         Constructor for the DepthwiseSeparableConv class.
        
#         Args:
#             in_channels (int): The number of input channels.
#             out_channels (int): The number of output channels.
#             stride (int, optional): The stride of the convolution. Defaults to 1.
#         """
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
#         # 
#         # The depthwise convolutional layer.
        
#         # Applies a depthwise convolution with a filter size of 3x3 and a stride of 'stride'.
#         # The number of input and output channels is equal to 'in_channels' and 'groups' is set to 'in_channels' to apply the convolution channel-wise.
#         # 
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         # 
#         # The pointwise convolutional layer.
        
#         # Applies a 1x1 convolution to reduce the number of channels to 'out_channels'.
#         # 
#         self.bn = nn.BatchNorm2d(out_channels)
#         # 
#         # The batch normalization layer.
        
#         # Normalizes the output of the convolutional layer to have zero mean and unit variance.
#         # 
#         self.relu = nn.ReLU(inplace=True)
#         # 
#         # The ReLU activation function.
        
#         # Applies the ReLU activation function to the output of the batch normalization layer.
        
#     def forward(self, x):
#         """
#         The forward pass of the DepthwiseSeparableConv layer.
        
#         Applies the depthwise convolutional layer, the pointwise convolutional layer, the batch normalization layer and the ReLU activation function to the input 'x'.
#         """
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.relu(x)


# class LightweightMTLNet224(nn.Module):
#     def __init__(self, num_classes_gender=2, num_classes_age=9, num_classes_ethnicity=7):
#         super(LightweightMTLNet224, self).__init__()

#         # Input layer
#         self.conv1 = DepthwiseSeparableConv(3, 64, stride=2)  
#         self.conv2 = DepthwiseSeparableConv(64, 128, stride=1) 
#         self.conv3 = DepthwiseSeparableConv(128, 128, stride=2)
#         self.conv4 = DepthwiseSeparableConv(128, 256, stride=1) 
#         self.conv5 = DepthwiseSeparableConv(256, 256, stride=2)
#         self.conv6 = DepthwiseSeparableConv(256, 512, stride=1) 
#         self.conv7 = DepthwiseSeparableConv(512, 512, stride=2)

#         # Pooling layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         # Task-specific classifiers
#         self.fc = nn.Linear(512, 256)

#         self.gender_fc = nn.Linear(256, num_classes_gender)
#         self.age_fc = nn.Linear(256, num_classes_age)
#         self.ethnicity_fc = nn.Linear(256, num_classes_ethnicity)

#     def forward(self, x):
#         # Pass through the convolutional layers
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)

#         # Global average pooling
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

#         # Fully connected layers
#         x = F.relu(self.fc(x),inplace=False) # Added ReLU activation

#         # Multi-task outputs
#         gender = self.gender_fc(x)
#         age = self.age_fc(x)
#         ethnicity = self.ethnicity_fc(x)

#         return gender, age, ethnicity

class MultitaskMobileNet(nn.Module):
    def __init__(self, num_gender_classes, num_age_classes, num_ethnicity_classes):
        super(MultitaskMobileNet, self).__init__()
        # Load pretrained MobileNet
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Remove the last classification layer
        self.features = self.mobilenet.features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.last_channel = self.mobilenet.last_channel

        # Task-specific heads
        self.gender_head = nn.Linear(self.last_channel, num_gender_classes)
        self.age_head = nn.Linear(self.last_channel, num_age_classes)
        self.ethnicity_head = nn.Linear(self.last_channel, num_ethnicity_classes)

    def forward(self, x):
        # Shared feature extraction
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Task-specific outputs
        gender_out = self.gender_head(x)
        age_out = self.age_head(x)
        ethnicity_out = self.ethnicity_head(x)

        return gender_out, age_out, ethnicity_out

num_age_classes = 9
num_ethnicity_classes = 7

@st.cache_resource
def load_model() -> torch.nn.Module:
    """
    Loads a pre-trained CNN model

    Returns:
        A pre-trained CNN model
    """
    # Load the pre-trained model
    model = MultitaskMobileNet(2, num_age_classes, num_ethnicity_classes).to(device)  

    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set the model to evaluation mode
    model.eval()

    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_to_gender = {0: 'Male', 1: 'Female'}
label_to_age = {0: '0-2', 1: '3-9', 2: '10-19', 3: '20-29', 4: '30-39', 5: '40-49', 6: '50-59', 7: '60-69', 8: 'more than 70'}
label_to_ethnicity = {0: 'White', 1: 'Black', 2: 'Latino_Hispanic', 3: 'East Asian', 4: 'Southeast Asian', 5: 'Indian', 6: 'Middle Eastern'}

def predict_proba(model, image, transform, device):
    """
    Predict the probability of the gender, age and ethnicity from an image

    Args:
        model (torch.nn.Module): The pre-trained model
        image (PIL.Image): The input image
        transform (torchvision.transforms.Compose): The transformation to apply to the image
        device (torch.device): The device to use for the prediction

    Returns:
        A dictionary with the probability of each class for gender, age and ethnicity
    """
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get the output of the model
        gender_output, age_output, ethnicity_output = model(image)

    # Get the probability of each class
    gender_proba = torch.softmax(gender_output, dim=1).cpu().numpy()[0]
    age_proba = torch.softmax(age_output, dim=1).cpu().numpy()[0]
    ethnicity_proba = torch.softmax(ethnicity_output, dim=1).cpu().numpy()[0]

    # Return the probability as a dictionary
    return {
        'gender_proba': gender_proba,
        'age_proba': age_proba,
        'ethnicity_proba': ethnicity_proba
    }


def plot_results(prediction_result, image):
    """
    Plot the probability of gender, age, and ethnicity from the model output

    Args:
        prediction_result (dict): The output of the model, which contains the probability of each class for gender, age and ethnicity
        image (PIL.Image): The input image

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(60, 20))
    
    # Plot gender probability
    labels = ['Male', 'Female']
    axes[0].bar(labels, prediction_result['gender_proba'])
    axes[0].tick_params(axis='x', rotation=45) # Rotate x-axis labels
    axes[0].set_title('Gender Probability')
    axes[0].set_ylim([0, 1])  

    # Plot age probability
    labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
    axes[1].bar(labels, prediction_result['age_proba'])
    axes[1].tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability
    axes[1].set_title('Age Probability')
    axes[1].set_ylim([0, 1])

    # Plot ethnicity probability
    labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
    axes[2].bar(labels, prediction_result['ethnicity_proba'])
    axes[2].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    axes[2].set_title('Ethnicity Probability')
    axes[2].set_ylim([0, 1])

    # Increase font size of x-axis labels
    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(50)

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Render in Streamlit
    st.pyplot(fig)

# Streamlit app
st.title("Gender, Age, and Ethnicity Classification")
# st.markdown("<h1 style='text-align: center;'>Gender, Age, and Ethnicity Classification</h1>", unsafe_allow_html=True)

st.sidebar.title('Navigation')
selection = st.sidebar.radio('',['Gender, Age, Ethnicity Classifier','Model Description','Author'])

if selection == 'Gender, Age, Ethnicity Classifier':
    # st.header("Gender, Age, and Ethnicity Classification")

    st.header("Instruction")

    st.write("Upload an image to predict gender, age group, and ethnicity.")

    st.write("Upload an image of a face to predict gender, age group, and ethnicity. The image should contain only one face (similar to the example images below). The model will classify the gender, and the age group, and ethnicity of the person in the image.")

    st.subheader("Example Inputs")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("images/image1.jpg")
    with col2:
        st.image("images/image2.jpg")
    with col3:
        st.image("images/image3.jpg")
    with col4:
        st.image("images/image4.jpg")
    

    st.write("This app is built using Streamlit and PyTorch.")

    st.header("Upload Image")
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg","webp"])

    if uploaded_image is not None:
        # image_cv = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        image = Image.open(uploaded_image)
        face_image = detect_face(image)

        if face_image is None:
            st.write("No face detected in the uploaded image. Please upload another image.")
            st.stop()

        prediction_result = predict_proba(model, face_image, transform, device)

        col1 , col2 = st.columns(2)
        with col1:
            st.write("**Uploaded Image**")
            st.image(image, caption='Uploaded Image',width = 200)
        with col2:
            st.write("**Detected Face**")
            st.image(face_image, caption='Detected Face',width = 200)

        gender_label = label_to_gender[np.argmax(prediction_result['gender_proba'])]
        age_label = label_to_age[np.argmax(prediction_result['age_proba'])]
        ethnicity_label = label_to_ethnicity[np.argmax(prediction_result['ethnicity_proba'])]

        st.write(f"**Predicted Labels:**")
        st.write(f"- Gender: {gender_label}")
        st.write(f"- Age group: {age_label}")
        st.write(f"- Ethnicity: {ethnicity_label}")

        st.write("**Probabilities of the predictions:**")
        plot_results(prediction_result, image)
        # st.pyplot(plt)

elif selection == 'Model Description':
    st.header("Model Description")

    st.write("The model is trained using a multi-task learning approach to predict gender, age, and ethnicity simultaneously. It is trained on the FairFace dataset, which provides diverse images representing various demographics.")

    st.write("Built on MobileNet, a lightweight and efficient CNN, the model processes images through multiple layers to extract and refine features. The final layers are specialized for accurate predictions of gender, age, and ethnicity.")

    # st.write("The model is built on MobileNet, a lightweight and efficient CNN. It extracts features through multiple layers, progressively refining them to identify patterns. The final layers are specialized to predict gender, age, and ethnicity.")

    # st.write("The model is trained using the Adam optimization algorithm and the cross-entropy loss function. The Adam optimization algorithm is a type of stochastic gradient descent algorithm that is used to optimize the model's parameters. The cross-entropy loss function is a measure of the difference between the predicted output and the actual output. The model is trained to minimize the cross-entropy loss function, which means that the model is trained to predict the correct output for a given input.")

    st.write("The code for the model is available [here](https://github.com/u1kemp/GenderAgeEthnicityClassification).")
    
elif selection == 'Author':
    st.header("Author")

    st.write("This app was built by [Utpalraj Kemprai](https://u1kemp.github.io/index.html) as a personal project on Multi-task Learning and CNN. The code for this app and model is available [here](https://github.com/u1kemp/GenderAgeEthnicityClassification).")

    st.write("I'm always open to discussing topics like data science, machine learning, and deep learning. Feel free to reach out via email, which I check regularly, or connect with me on LinkedIn.")
