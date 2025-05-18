from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def carregar_modelo(model_path):
    print("Carregando tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    print("Carregando modelo base...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')
        print("Usando GPU")
    else:
        print("Usando CPU")

    return tokenizer, model

def gerar_resposta(model, tokenizer, prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta

def main():
    model_path = "C:/Users/Leandro/OneDrive/Documentos/estudos/ALice.IA/Alice-rodada3"

    tokenizer, model = carregar_modelo(model_path)

    print("Alice está pronta! Digite 'sair' para encerrar.")

    while True:
        prompt = input("\nVocê: ")
        if prompt.lower() == "sair":
            print("Encerrando...")
            break

        resposta = gerar_resposta(model, tokenizer, prompt)
        print(f"Alice: {resposta}")

if __name__ == "__main__":
    main()
