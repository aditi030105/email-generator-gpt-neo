import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Load the tokenizer and model (ensure this runs once at the top)
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token  # Important fix!
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.set_page_config(page_title="Email Generator", page_icon="✉️")
st.title("📧 GPT-Neo Email Generator")

recipient = st.text_input("👤 Recipient's Name")
event = st.text_input("🎯 Event or Occasion (e.g., AI Summit 2025)")
instructions = st.text_area("📝 Extra Info (e.g., topic, location, time)")

if st.button("Generate Email", key="generate_button"):
    if not recipient or not event:
        st.warning("⚠️ Please fill in the recipient name and event.")
    else:
        # Better structured prompt
        prompt = (
            f"Write a professional and concise email.\n\n"
            f"To: {recipient}\n"
            f"Subject: Invitation to {event}\n\n"
            f"Body:\n"
            f"Dear {recipient},\n\n"
            f"I hope this message finds you well. I am writing to invite you to {event}, "
            f"{instructions}\n\n"
            f"Best regards,\nYour Name\n"
            f"---\n(Do not add anything after this.)"
        )

        with st.spinner("Generating your email..."):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            output = model.generate(
                input_ids,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            email = decoded.split('---')[0].strip()  # Trim any noise after the separator

        st.subheader("✉️ Generated Email")
        st.write(email)

        st.success("✅ Email generated successfully!")
