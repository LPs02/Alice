import torch
from sentence_transformers import SentenceTransformer
from Alice_core import LinfModel, load_model
import torch.nn as nn
import torch.optim as optim
import re
from setup import get_system_prompt
from APIs import get_custom_response  # corrigido nome do arquivo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedder = SentenceTransformer("sentence-transformers/LaBSE", device=device)
model = load_model(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

history = []

SYSTEM_PROMPT = get_system_prompt()

def clean_response(text: str) -> str:
    return re.sub(r'^(Alice|Leandro|Isaque|Lorenna)\s*:\s*', '', text.strip(), flags=re.IGNORECASE)

def retrieve_best_response(output_embedding):
    # Mock temporário (pode implementar busca no dataset aqui)
    return "Alice: Hmm... isso me faz pensar. Pode me contar mais sobre isso?"

def process_input(user_id: str, input_text: str) -> str:
    # Primeiro verifica se input tem resposta especial (API)
    api_response = get_custom_response(input_text.lower())
    if api_response:
        return api_response

    # Monta contexto
    context = SYSTEM_PROMPT + "\n\n"
    if history:
        context += "\n".join(history[-6:]) + "\n"
    context += f"User: {input_text}"

    input_emb = embedder.encode(context, convert_to_tensor=True).to(device)

    with torch.no_grad():
        output_emb = model(input_emb)

    raw_response = retrieve_best_response(output_emb)
    response = clean_response(raw_response)

    # Meta-learning: precisa de resposta correta para treinar online
    # No exemplo, vamos assumir resposta gerada como "target" (cuidado!)
    target_emb = embedder.encode(response, convert_to_tensor=True).to(device)

    model.train()
    optimizer.zero_grad()
    pred_emb = model(input_emb)
    loss = loss_fn(pred_emb, target_emb)
    loss.backward()
    optimizer.step()
    model.eval()

    history.append(f"User: {input_text}")
    history.append(f"Alice: {response}")

    return response

if __name__ == "__main__":
    while True:
        user_input = input("Você: ")
        response = process_input("default", user_input)
        print("Alice:", response)
