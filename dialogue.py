import argparse
import warnings

from transformers import AutoTokenizer, AutoModelWithLMHead

warnings.filterwarnings('ignore')


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Type your prompt to get the response from the model",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def process_response(response):
    response = response.capitalize()
    response = response.replace("\n", " ")
    return response


def print_dialogue(prompt, response):
    print("— " + prompt)
    print("— " + response)


args = parse_args()

checkpoint_path = '/home/consent-flower/tinkoff/sirius-test-nlp-2023/model/checkpoint-5500'
base_model = 'tinkoff-ai/ruDialoGPT-medium'

device = 'cpu'
model = AutoModelWithLMHead.from_pretrained(checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model)

prompt = args.prompt
full_prompt = f'@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ {prompt} @@ВТОРОЙ@@'
inputs = tokenizer(full_prompt, return_tensors='pt')
generated_token_ids = model.generate(
    **inputs,
    top_k=10,
    top_p=0.95,
    num_beams=3,
    num_return_sequences=3,
    do_sample=True,
    no_repeat_ngram_size=2,
    temperature=1.2,
    repetition_penalty=1.2,
    length_penalty=1.0,
    eos_token_id=50257,
    max_new_tokens=40
)
context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]

response = context_with_response[2][len(full_prompt)+1:]
response = response[:response.find("<pad>")]
response = process_response(response)

print(response)
