# RAG-Based Scientific Chatbot

This project implements a Retrieval-Augmented Generation (RAG) Scientific Chatbot that provides insightful and contextually relevant responses to queries based on scientific documents. The chatbot leverages a language model to retrieve and synthesize information from a pre-constructed vector database of scientific knowledge.

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines information from a set of relevant documents to answer questions.
- **Experiment Tracking**: Follow ongoing experiments and log updates.

## Architecture

- **Frontend**: User interface for interaction.
- **Backend**: Handles data retrieval and processing using RAG architecture.
- **Database**: Stores scientific documents and experiment data.
- **Deployment**: The Streamlit app is deployed on an **Amazon EC2 instance** for scalable access.

## Requirements

- Python 3.7 or higher
- Streamlit (for the frontend)
- Transformers (for the language model)
- FAISS (for efficient similarity search)
- EC2 Gpu based instance to deploy the app and interact with the chatbot
- Install Nvidia 

## Setup within EC2 insatnce

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/rag-scientific-chatbot.git
   cd rag-scientific-chatbot
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash 
   pip install -r requirements.txt
   ```
4. Set up the environment variables if necessary (specify any required environment variables).

## Steps Taken throughtout the deployment phase:

1. Find the EC2 Instance’s Public DNS or IP Address and verify that the type is **g4dn.xlarge**

   <img width="732" alt="project1" src="https://github.com/user-attachments/assets/6220533b-ee67-488f-b69f-5f67e5382bb0">
   
2. Connect to the EC2 Instance via SSH
   ```bash
   ssh -i /path/to/your-key.pem ec2-user@your-public-dns
   ```
3. Accept the Security Warning : yes
4. You may need to set Permissions for the Key Pair as follows:
   ```bash
   chmod 400 /path/to/your-key.pem
   ```
5. You need to install CUDA Toolkit to utilize GPU resources for PyTorch.
   ```bash
   nvcc --version  # Check if nvcc is installed
   sudo apt install nvidia-cuda-toolkit  # Install CUDA Toolkit
   ```
   <img width="698" alt="project3" src="https://github.com/user-attachments/assets/ccc0560e-d7f6-4d8a-af94-b3e5b5374718">
   
6. After cloning the Project Repository inside the EC2 instance and installing the required packages you need to run the streamlit app file ```streamlit run app.py``` and acess the  **ngrok tunells** links or you can sart the application with **EC2 instance's public IP address**

   ![382511484-cb6b875d-6a0b-49bf-a27f-091125a579c7](https://github.com/user-attachments/assets/f0b037cf-da39-42ed-ae35-be9ba4f4db29)

   
7. Interact with the chatbot through the provided interface.

   ![cahtboot](https://github.com/user-attachments/assets/cffdfca8-08aa-4ebf-900f-6cc38c28c516)

8. The retrieved documents:

   <img width="558" alt="image" src="https://github.com/user-attachments/assets/b6f9b72a-22de-4766-9476-aa8f3a83e552"/>
   

   

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourFeature`)
3. Commit your Changes (`git commit -m 'Add some feature'`)
4. Push to the Branch (`git push origin feature/YourFeature`)
5. Open a Pull Request
   
## Contact

For further information or questions, feel free to reach out:
- **Email**: najmaelboutaheri@gmail.com
