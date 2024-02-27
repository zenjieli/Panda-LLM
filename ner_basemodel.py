from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Llama-2-base")
model = AutoModelForCausalLM.from_pretrained("Llama-2-base")

# Define the prompt with a few examples
prompt = """
[EXAMPLE]
Text: "John lives in New York."
Entities: {"John": "PERSON", "New York": "LOCATION"}

[EXAMPLE]
Text: "Microsoft was founded by Bill Gates."
Entities: {"Microsoft": "ORGANIZATION", "Bill Gates": "PERSON"}

[EXAMPLE]
Text: "The Eiffel Tower is in Paris."
Entities: {"Eiffel Tower": "LOCATION", "Paris": "LOCATION"}

[QUERY]
Text: "Apple was established in Cupertino."
"""

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate the output
output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# Decode the output
output = tokenizer.decode(output_ids[0])

print(output)
