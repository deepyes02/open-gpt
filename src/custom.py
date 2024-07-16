from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode input text
input_text = "I think that the world needs to be more careful of AI implementations, since it could turn dangerous"
inputs = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
outputs = model.generate(inputs, max_length=1000, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)