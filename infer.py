prompt = """Instruction: Generate a protein sequence.
Input: Length ~150 amino acids
Output:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=300,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
