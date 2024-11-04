
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
model_path = 'multi_task_model.pth' # to be updated
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DepthwiseSeparableConv(nn.Module):
    """
    A class implementing a depthwise separable convolutional layer.
    
    Depthwise separable convolution is a technique for reducing the number of parameters and computation required for a convolutional layer.
    It first applies a depthwise convolution (a convolution with a filter that is applied channel-wise) and then a pointwise convolution (a 1x1 convolution).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Constructor for the DepthwiseSeparableConv class.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int, optional): The stride of the convolution. Defaults to 1.
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        # 
        # The depthwise convolutional layer.
        
        # Applies a depthwise convolution with a filter size of 3x3 and a stride of 'stride'.
        # The number of input and output channels is equal to 'in_channels' and 'groups' is set to 'in_channels' to apply the convolution channel-wise.
        # 
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # 
        # The pointwise convolutional layer.
        
        # Applies a 1x1 convolution to reduce the number of channels to 'out_channels'.
        # 
        self.bn = nn.BatchNorm2d(out_channels)
        # 
        # The batch normalization layer.
        
        # Normalizes the output of the convolutional layer to have zero mean and unit variance.
        # 
        self.relu = nn.ReLU(inplace=True)
        # 
        # The ReLU activation function.
        
        # Applies the ReLU activation function to the output of the batch normalization layer.
        
    def forward(self, x):
        """
        The forward pass of the DepthwiseSeparableConv layer.
        
        Applies the depthwise convolutional layer, the pointwise convolutional layer, the batch normalization layer and the ReLU activation function to the input 'x'.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class LightweightMTLNet224(nn.Module):
    def __init__(self, num_classes_gender=2, num_classes_age=5, num_classes_ethnicity=7):
        """
        Initializes the LightweightMTLNet224 model.

        Args:
            num_classes_gender (int): Number of classes for gender classification.
            num_classes_age (int): Number of classes for age classification.
            num_classes_ethnicity (int): Number of classes for ethnicity classification.
        """
        super(LightweightMTLNet224, self).__init__()
        
        # Define the depthwise separable convolutional layers
        # Each layer reduces the spatial dimensions and increases the channel dimension
        self.conv1 = DepthwiseSeparableConv(3, 32, stride=2)  # Initial layer, outputs 112x112 feature map
        self.conv2 = DepthwiseSeparableConv(32, 64, stride=2)  # Outputs 56x56 feature map
        self.conv3 = DepthwiseSeparableConv(64, 128, stride=2) # Outputs 28x28 feature map
        self.conv4 = DepthwiseSeparableConv(128, 256, stride=2) # Outputs 14x14 feature map
        self.conv5 = DepthwiseSeparableConv(256, 512, stride=2) # Final convolutional layer, outputs 7x7 feature map
        
        # Global average pooling layer to reduce the feature map to a single 1x1 spatial size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define task-specific fully connected layers for classification
        # These layers output the final predictions for each task
        self.gender_fc = nn.Linear(512, num_classes_gender)    # Fully connected layer for gender classification
        self.age_fc = nn.Linear(512, num_classes_age)          # Fully connected layer for age classification
        self.ethnicity_fc = nn.Linear(512, num_classes_ethnicity) # Fully connected layer for ethnicity classification

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing the image data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Outputs for gender, age, and ethnicity classification.
        """
        # Sequentially pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Apply global average pooling to reduce the spatial dimensions
        x = self.avgpool(x)
        
        # Flatten the pooled feature map for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Compute the outputs for each classification task
        gender = self.gender_fc(x)    # Output for gender classification
        age = self.age_fc(x)          # Output for age classification
        ethnicity = self.ethnicity_fc(x) # Output for ethnicity classification
        
        # Return the outputs as a tuple
        return gender, age, ethnicity


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
    model = LightweightMTLNet224(2, num_age_classes, num_ethnicity_classes).to(device)  

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
    # axes[0].imshow(image)
    # axes[0].set_title("Image")
    # axes[0].axis('off')

    # Plot gender probability
    labels = ['Male', 'Female']
    axes[0].bar(labels, prediction_result['gender_proba'])
    axes[0].set_title('Gender Probability')

    # Plot age probability
    labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
    axes[1].bar(labels, prediction_result['age_proba'])
    axes[1].tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability
    axes[1].set_title('Age Probability')

    # Plot ethnicity probability
    labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
    axes[2].bar(labels, prediction_result['ethnicity_proba'])
    axes[2].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    axes[2].set_title('Ethnicity Probability')

    # Increase font size of x-axis labels
    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(50)

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.show()


# Streamlit app
st.title("Gender, Age, and Ethnicity Classification")
# st.markdown("<h1 style='text-align: center;'>Gender, Age, and Ethnicity Classification</h1>", unsafe_allow_html=True)

st.sidebar.title('Navigation')
selection = st.sidebar.radio('',['Gender, Age, Ethnicity Classifier','Model Description','Author'])

if selection == 'Gender, Age, Ethnicity Classifier':
    # st.header("Gender, Age, and Ethnicity Classification")

    st.header("Instruction")

    st.write("Upload an image to predict gender, age group, and ethnicity.")

    st.write("This web app is trained on the fairface dataset, which contains images of people with different genders, age groups, and ethnicities. The model is trained using the multi-task learning approach. Upload an image to predict its gender, age group, and ethnicity. The image should contain only one face and should contain minimal background (similar to the example images given below). The model will classify the gender, and the age group, and ethnicity of the person in the image.")

    st.subheader("Example Inputs")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("images/image1.png")
    with col2:
        st.image("images/image2.png")
    with col3:
        st.image("images/image3.png")
    with col4:
        st.image("images/image4.png")
    

    st.write("This app is built using Streamlit and PyTorch.")

    st.header("Upload Image")
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        prediction_result = predict_proba(model, image, transform, device)
        st.image(image, caption='Uploaded Image')

        gender_label = label_to_gender[np.argmax(prediction_result['gender_proba'])]
        age_label = label_to_age[np.argmax(prediction_result['age_proba'])]
        ethnicity_label = label_to_ethnicity[np.argmax(prediction_result['ethnicity_proba'])]

        st.write(f"**Predicted Labels:**")
        st.write(f"Gender: {gender_label}")
        st.write(f"Age group: {age_label}")
        st.write(f"Ethnicity: {ethnicity_label}")

        st.write("*Probabilities of the predictions:*")
        plot_results(prediction_result, image)
        st.pyplot(plt)

elif selection == 'Model Description':
    st.header("Model Description")

    st.write("The model is trained using the multi-task learning approach. The model was trained on the fairface dataset, which contains images of people with different genders, ages, and ethnicities.")

    st.write("The model is a multi-task learning model, which is a type of deep learning model that is trained on multiple tasks simultaneously. The model is trained on the fairface dataset, which contains images of people with different genders, ages, and ethnicities. The model is trained to predict the gender, age, and ethnicity of the person in the image.")

    st.write("The model uses a convolutional neural network (CNN) architecture. The CNN is a type of deep learning model that is used to process images and other two-dimensional data. The CNN is composed of multiple layers, each of which processes the input data in a different way. The layers are composed of multiple filters, which are used to extract features from the input data. The features are then passed on to the next layer, where they are processed further. The output of the final layer is the prediction of the gender, age, and ethnicity of the person in the image.")

    st.write("The model is trained using the Adam optimization algorithm and the cross-entropy loss function. The Adam optimization algorithm is a type of stochastic gradient descent algorithm that is used to optimize the model's parameters. The cross-entropy loss function is a measure of the difference between the predicted output and the actual output. The model is trained to minimize the cross-entropy loss function, which means that the model is trained to predict the correct output for a given input.")

    st.write("The code for the model is available [here](https://github.com/u1kemp/GenderAgeEthnicityClassification).")
    
elif selection == 'Author':
    st.header("Author")

    st.write("This app was built by [Utpalraj Kemprai](https://u1kemp.github.io/index.html) as a personal project on CNN. The code for this app and model is available [here](https://github.com/u1kemp/GenderAgeEthnicityClassification).")

    st.write("I am free to discuss about academics or research. Feel free to reach out to me through [email](mailto:utplarajkemprai2001@gmail.com), as I keep on checking emails regularly. You can also contact me via LinkedIn.")
