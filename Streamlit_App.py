import subprocess
from pyngrok import ngrok


# Run Streamlit app
def run_streamlit():
    process = subprocess.Popen(['streamlit', 'run', 'app_chatbot.py'])
    return process

# Establish ngrok tunnel
def start_ngrok():
    public_url = ngrok.connect(8501)
    print(f"Streamlit app is live at: {public_url}")
    return public_url

# Start Streamlit and ngrok
process = run_streamlit()
url = start_ngrok()
