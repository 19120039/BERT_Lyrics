
import streamlit as st
from transformers import BertTokenizer, BertForMaskedLM
import torch

model = BertForMaskedLM.from_pretrained('PUSH/lyrics_train')
tokenizer = BertTokenizer.from_pretrained('PUSH/lyrics_train')


def main():
    st.title("Song Lyrics Generator")

    seed_text = st.text_area("Enter Seed Text", value="I love you", height=100)

    if st.button("Generate Lyrics"):
        # Set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Tokenize the seed text
        input_ids = tokenizer.encode(seed_text, add_special_tokens=True, return_tensors="pt").to(device)

        # Generate lyrics
        model.eval()  # Set the model to evaluation mode
        generated_lyrics = model.generate(
            input_ids=input_ids,
            max_length=100,
            num_return_sequences=6,
            temperature=0.8,
            do_sample=True,
        )

        # Decode and display the generated lyrics
        for i, generated_sequence in enumerate(generated_lyrics):
            generated_sequence = generated_sequence.tolist()
            decoded_lyrics = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            st.write(f"Generated Lyrics {i+1}:")
            st.write(decoded_lyrics)


if __name__ == "__main__":
    main()
