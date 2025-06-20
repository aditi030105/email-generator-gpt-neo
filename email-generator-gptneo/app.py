import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import re

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.set_page_config(page_title="Email Generator", page_icon="✉️")
st.title("📬 GPT-Neo Email Generator")

recipient = st.text_input("👤 Recipient's Name")
event = st.text_input("📌 Event or Reason")
instructions = st.text_area("📝 Extra Details (Date, Topic, Place, etc.)")

def clean_output(text):
    text = text.strip().replace("\n\n", "\n")
    lines = text.split("\n")
    seen = set()
    final_lines = []
    for line in lines:
        if line.strip() and line.strip() not in seen:
            final_lines.append(line.strip())
            seen.add(line.strip())
    cleaned = "\n".join(final_lines)
    return cleaned

if st.button("Generate Email", key="generate_btn"):
    if not recipient or not event:
        st.warning("⚠️ Please enter both recipient and event.")
    else:
        with st.spinner("⏳ Generating email..."):
            prompt = (
                f"Write a formal, polite and complete email to Dr. {recipient} about {event}. "
                f"The email should include these details: {instructions}.\n\n"
                f"Start the email like this:\n"
                f"Dear Dr. {recipient},\n"
            )

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            try:
                output = model.generate(
                    input_ids,
                    max_length=350,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_email = decoded[len(prompt):].strip()

                st.subheader("✉️ Generated Email")
                st.write(f"Dear Dr. {recipient},\n\n" + clean_output(generated_email))
                st.success("✅ Email generated successfully!")

            except Exception as e:
                st.error(f"❌ Error generating email: {e}")
