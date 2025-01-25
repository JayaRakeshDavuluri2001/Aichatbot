
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
import torch

# Flask app setup
app = Flask(__name__)

# Load the pre-trained DialoGPT-small model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history_ids = None  # Keeps track of the conversation
user_name = None  # Stores the user's name for personalization

# Function to get the current timestamp
def get_timestamp():
    return datetime.datetime.now().strftime("%H:%M:%S")

@app.route("/")
def index():
    return render_template("index.html")  # Render the chatbot page

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history_ids, user_name

    # Get user input
    user_input = request.json.get("message", "").strip()

    # Greet and ask for the user's name 
    if user_name is None:
        if user_input.lower() in ["hi", "hello", "hey", "hola"]:
            response = f"Hello! What's your name? ({get_timestamp()})"
            return jsonify({"response": response})
        elif user_input:
            user_name = user_input  # Save the user's name
            response = f"Nice to meet you, {user_name}! How can I assist you today? ({get_timestamp()})"
            return jsonify({"response": response})

    # Predefined intents
    if "weather" in user_input.lower():
        response = f"I can’t provide live weather updates, {user_name}, but you can check a weather app! ({get_timestamp()})"
    elif "time" in user_input.lower():
        current_time = get_timestamp()
        response = f"The current time is {current_time}, {user_name}."
    elif "." in user_input.lower():
        response = f"I’m sorry, I didn’t understand that. Could you please rephrase? ({get_timestamp()})"
    else:
        # If no predefined intent, generate a response using DialoGPT
        try:
            # Tokenize user input and append it to chat history
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            chat_history_ids = (
                new_user_input_ids
                if chat_history_ids is None
                else torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            )

            # Generate a response
            bot_output = model.generate(
                chat_history_ids,
                max_length=1000,  # Limit response length
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_response = tokenizer.decode(bot_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

            # Add timestamp to the response
            response = f"{generated_response} ({get_timestamp()})"

        except Exception as e:
            response = f"I’m sorry, I didn’t understand that. Could you please rephrase? ({get_timestamp()})"

    # Return the chatbot's response
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
