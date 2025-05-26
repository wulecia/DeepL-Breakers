
import torch
from transformers import AutoTokenizer
from model_utils import get_model

def main():
    # === Settings ===
    model_path = "best_Berkeley_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # === Load model ===
    model = get_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # === Example sentences ===
    sentences = [
        "I hate you and your people.",
        "We should celebrate our differences and be kind.",
        "Immigrants are ruining this country.",
        "Hairy women are scary.",
    ]

    # === Tokenize ===
    inputs = tokenizer(
        sentences,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # === Predict ===
    with torch.no_grad():
        preds_num, preds_bin = model(input_ids, attention_mask)

    # === Format predictions ===
    preds_bin_binary = (preds_bin > 0.5).int()

    # === Print results ===
    for i, sentence in enumerate(sentences):
        print(f"\n Sentence: {sentence}")
        print(f"  Regression: {preds_num[i].cpu().numpy().round(3)}")
        print(f"  Binary (probs): {preds_bin[i].cpu().numpy().round(3)}")
        print(f"  Binary (0/1):   {preds_bin_binary[i].cpu().numpy()}")


if __name__ == "__main__":
    main()
