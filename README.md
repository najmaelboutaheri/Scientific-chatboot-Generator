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
- Flask (for the backend)
- Streamlit (for the frontend)
- Transformers (for the language model)
- FAISS (for efficient similarity search)
- EC2 Gpu based instance to deploy the app and interact with the chatbot

## Setup

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
   
## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the **EC2 instance's public IP address**to access the app.
3. Interact with the chatbot through the provided interface.
   
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
