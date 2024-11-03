
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
# Assuming 'multi_task_inceptionv3.pth' is in the same directory
model_path = 'multi_task_inceptionv3.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    """
    A multi-task CNN model for age, gender, and ethnicity classification

    Attributes:
        input_size (int): The size of the input image
        num_age_classes (int): The number of classes for age classification
        num_ethnicity_classes (int): The number of classes for ethnicity classification
    """

    def __init__(self, input_size, num_age_classes, num_ethnicity_classes ):
        """
        Initializes the model

        Args:
            input_size (int): The size of the input image
            num_age_classes (int): The number of classes for age classification
            num_ethnicity_classes (int): The number of classes for ethnicity classification
        """
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3,padding=0)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=6,padding=0)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=6,padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(8*4*4, 200)
        self.fc2 = nn.Linear(200, 10)
        
        # Task-specific output layers
        self.gender_fc = nn.Linear(10, 2)              # Output for gender classification (2 classes)
        self.age_fc = nn.Linear(10, num_age_classes)   # Output for age classification
        self.ethnicity_fc = nn.Linear(10, num_ethnicity_classes)  # Output for ethnicity classification

    def forward(self, x, verbose=False):
        """
        Perform a forward pass of the model

        Args:
            x (torch.Tensor): Input tensor
            verbose (bool): Flag to print intermediate outputs for debugging

        Returns:
            tuple: Outputs for gender, age, and ethnicity classification
        """
        # Pass through first convolutional layer and ReLU activation
        x = self.conv1(x)
        x = F.relu(x)

        # Pass through second convolutional layer and ReLU activation
        x = self.conv2(x)
        x = F.relu(x)
        
        # Apply max pooling
        x = F.max_pool2d(x, kernel_size=2)

        # Pass through third convolutional layer and ReLU activation
        x = self.conv3(x)
        x = F.relu(x)

        # Apply max pooling
        x = F.max_pool2d(x, kernel_size=2)

        # Flatten the tensor for fully connected layers
        x = x.view(-1, 8*4*4)

        # Pass through first fully connected layer and ReLU activation
        x = self.fc1(x)
        x = F.relu(x)

        # Pass through second fully connected layer
        x = self.fc2(x)

        # Task-specific outputs with log softmax activation
        gender_output = F.log_softmax(self.gender_fc(x), dim=1)
        age_output = F.log_softmax(self.age_fc(x), dim=1)
        ethnicity_output = F.log_softmax(self.ethnicity_fc(x), dim=1)

        return gender_output, age_output, ethnicity_output


num_age_classes = 9
num_ethnicity_classes = 7
# model = CNN(28 * 28, num_age_classes, num_ethnicity_classes).to(device)  # Replace CNN with your actual model class
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

@st.cache_resource
def load_model() -> torch.nn.Module:
    """
    Loads a pre-trained CNN model

    Returns:
        A pre-trained CNN model
    """
    # Load the pre-trained model
    model = CNN(28 * 28, num_age_classes, num_ethnicity_classes).to(device)  

    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set the model to evaluation mode
    model.eval()

    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
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
            item.set_fontsize(40)

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
    st.header("Gender, Age, and Ethnicity Classification")

    st.header("Instruction")

    st.write("Upload an image to predict gender, age, and ethnicity.")

    st.write("This web app is trained on the fairface dataset, which contains images of people with different genders, ages, and ethnicities. The model is trained using the multi-task learning approach. Upload an image to predict its gender, age, and ethnicity. The image should contain only one face and should contain minimal background (similar to a person's photo on a ID card or driver's license). The model will classify the image as either male or female, and the age and ethnicity of the person in the image.")
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
        st.write(f"Age: {age_label}")
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

    st.write("The code for the model is available [here](https://github.com/u1kemp/git checkout main
    git merge master
    git push origin main).")
    
elif selection == 'Author':
    st.header("Author")

    st.write("This app was built by [Utpalraj Kemprai](https://u1kemp.github.io/index.html) as a personal project on CNN. The code for this app and model is available [here](https://github.com/u1kemp/GenderAgeEthnicityClassification).")

    st.write("I am free to discuss about academics or research. Feel free to reach out to me through [email](mailto:utplarajkemprai2001@gmail.com), as I keep on checking emails regularly. You can also contact me via LinkedIn.")
