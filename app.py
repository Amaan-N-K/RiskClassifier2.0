import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertModel, BertTokenizer
import os
import google.generativeai as genai

app = Flask(__name__)
os.environ["GEMINI_API_KEY"] = 'AIzaSyCb5ebBaHZH-20alyFC_CK5kmm01S8Fq4c'

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = gen_model.start_chat(history=[])

MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-05


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


# Initialize the model
model = BERTClass()

# Load the checkpoint
checkpoint = torch.load('best_model2.pt', map_location=torch.device('cpu'))

# Extract the model state dictionary, ignoring missing keys
model_state_dict = checkpoint['state_dict']
missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

# Set the model to evaluation mode
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess(description):
    encodings = tokenizer.encode_plus(
        description,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = encodings['input_ids'].to(device, dtype=torch.long)
        attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)
        output = model(input_ids, attention_mask, token_type_ids)
        final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()

    return final_output


def postprocess(output):
    final_output = output[0]  # Extract the inner list
    risks = ["Cybersecurity Threats", "Data Loss", "IT Failures", "Third-Party Vendor Risks"]
    sorted_indices = np.argsort(final_output)[::-1]  # Sort indices in descending order
    sorted_risks = [risks[i] for i in sorted_indices]
    return sorted_risks


def explain_and_mitigate(risks, description):
    print("here")
    # Construct the prompt for the LLM
    explanation_prompt = f"Explain how the following description relates to each risk, in order of the risks ranked. Each explanation of the risks should be separated by a line and should be max 100 words each.\n\nDescription: {description}\n\nRisks:\n"
    for risk in risks:
        explanation_prompt += f"Risk: {risk}\nExplanation:\n\n"

    mitigation_prompt = "Generate a mitigation plan for each risk, also 100 words each:\n\n"
    for risk in risks:
        mitigation_prompt += f"Risk: {risk}\nMitigation Plan:\n\n"

    combined_prompt = explanation_prompt + "\n" + mitigation_prompt

    # Send the prompt to the Gemini API and receive the response
    response = chat_session.send_message(combined_prompt)

    # Extract the response text
    response_text = response.text.strip()

    # Separate explanations and mitigation plans
    explanations = []
    mitigations = []

    parts = response_text.split("\n\n")
    explanation_part = True
    for part in parts:
        if "Mitigation Plan:" in part:
            explanation_part = False

        if explanation_part:
            explanations.append(part)
        else:
            mitigations.append(part)

    explanations_text = "\n\n".join(explanations)
    mitigations_text = "\n\n".join(mitigations)

    return explanations_text, mitigations_text


def predict2():
    description = "Upgrading all internal communication systems to secure messaging platforms."
    output = preprocess(description)
    print(output)
    sorted_risks = postprocess(output)
    print(sorted_risks)
    explanation, mitigation_plan = explain_and_mitigate(sorted_risks, description)
    print(explanation)
    print(mitigation_plan)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Print for debugging
    print("first")
    data = request.get_json()
    description = data.get('description')
    print("Request JSON data:", data)
    print("Description:", description)

    output = preprocess(description)
    sorted_risks = postprocess(output)
    explanation, mitigation_plan = explain_and_mitigate(sorted_risks, description)

    return jsonify({
        'risks': sorted_risks,
        'explanation': explanation,
        'mitigation_plan': mitigation_plan
    })

if __name__ == "__main__":
    app.run(debug=True)