# Importing necessary libraries
import time
import random
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set page title
st.set_page_config(
    page_title="ChatGPT-124M",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖",
)

# Title
st.title("🤖 ChatGPT-124M")

# --- Initialize Session State with Defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "max_length" not in st.session_state:
    st.session_state.max_length = 50

if "do_sample" not in st.session_state:
    st.session_state.do_sample = True

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "top_p" not in st.session_state:
    st.session_state.top_p = 0.95

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.9

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load model and tokenizer
MODEL_NAME = "GPT_124M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)


# Function to convert a string into a generator (Fake stream output)
def string_to_generator(text):
    """Yields text one character at a time for a streaming effect."""
    for char in text:
        time.sleep(0.005)
        yield char


# --- UI Controls for Generation Parameters ---
st.sidebar.header("⚙️ Generation Settings")


# Slider for max length (1 to 100)
max_length = st.sidebar.slider(
    "Max Length", min_value=1, max_value=100, key="max_length"
)

# Toggle for `do_sample`
do_sample = st.sidebar.toggle(
    "Enable Sampling", key="do_sample"
)  # If `do_sample` is enabled, enable additional parameters

# Slider for top k (1 to 100)
top_k = st.sidebar.slider(
    "Top-K", min_value=1, max_value=100, disabled=not do_sample, key="top_k"
)

# Slider for top p (0 to 1)
top_p = st.sidebar.slider(
    "Top-P",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    disabled=not do_sample,
    key="top_p",
)

# Slider for temperature (0 to 1)
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    disabled=not do_sample,
    key="temperature",
)

# Reset Generation Settings
if st.sidebar.button("Reset"):
    for st_key in [
        "messages",
        "do_sample",
        "max_length",
        "top_k",
        "top_p",
        "temperature",
    ]:
        del st.session_state[st_key]
    st.rerun()

# List of dynamic loading messages
loading_messages = [
    "Generating your response, please wait...",
    "Working on your response...",
    "Processing, this will just take a moment...",
    "Creating your response, hold on...",
    "Loading your answer, please be patient...",
]

# --- Chat Input ---
if prompt := st.chat_input(
    "The Earth revolves around", max_chars=400, key="chat_input"
):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        tokens = tokenizer.encode(prompt, return_tensors="pt")

        with st.spinner(random.choice(loading_messages)):
            generated_tokens = model.generate(
                tokens,
                max_length=max_length,
                do_sample=do_sample,
                top_k=top_k if do_sample else 1,
                top_p=top_p if do_sample else 1.0,
                temperature=temperature if do_sample else 1.0,
            )

            response_text = tokenizer.decode(generated_tokens)
            response = st.write_stream(string_to_generator(response_text))

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
