import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # <-- Adicionado
from collections import deque
import json

# Conversões binárias
def text_to_binary(text):
    return ' '.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    return ''.join(chr(int(b, 2)) for b in binary.split())

def text_to_tensor(text):  # <-- Adicionado
    binary = text_to_binary(text).replace(" ", "")
    array = [int(b) for b in binary]
    while len(array) < 512:
        array.append(0)
    array = array[:512]
    return torch.tensor(array, dtype=torch.float32).unsqueeze(0)

# Memória Semântica (conceitos e significados)
class SemanticMemory:
    def __init__(self):
        self.knowledge = {}  # conceito -> padrão binário
        self.cooccurrence = {}  # conceito -> { conceito_vizinho: peso }

    def store(self, concept, binary_pattern):
        self.knowledge[concept] = binary_pattern
        if concept not in self.cooccurrence:
            self.cooccurrence[concept] = {}

    def retrieve(self, concept):
        return self.knowledge.get(concept, None)

    def add_cooccurrence(self, concept1, concept2):
        if concept1 == concept2:
            return
        if concept1 not in self.cooccurrence:
            self.cooccurrence[concept1] = {}
        self.cooccurrence[concept1][concept2] = self.cooccurrence[concept1].get(concept2, 0) + 1

    def learn_from_interaction(self, input_text):
        tokens = input_text.split()
        for token in tokens:
            if token not in self.knowledge:
                self.store(token, text_to_binary(token))
        # Atualiza coocorrências entre todos tokens da frase
        for i, token1 in enumerate(tokens):
            for j, token2 in enumerate(tokens):
                if i != j:
                    self.add_cooccurrence(token1, token2)

    def generate_associations(self, concept, top_n=3):
        if concept not in self.cooccurrence:
            return []
        related = sorted(self.cooccurrence[concept].items(), key=lambda x: -x[1])
        return [rel[0] for rel in related[:top_n]]

    def generate_phrase(self, input_text, emotion_state=None):
        tokens = input_text.split()
        phrase_parts = []
        for token in tokens:
            phrase_parts.append(token)
            associates = self.generate_associations(token)
            if associates:
                phrase_parts.append("que me lembra " + ", ".join(associates))
        base_phrase = ". ".join(phrase_parts)
    
        if emotion_state:
            emotion, intensity = emotion_state
            if emotion == "alegria" and intensity > 0.3:
                base_phrase += " Isso me deixa feliz!"
            elif emotion == "tristeza" and intensity > 0.3:
                base_phrase += " Isso me deixa pensativa."
            elif emotion == "raiva" and intensity > 0.3:
                base_phrase += " Isso me incomoda."
            elif emotion == "surpresa" and intensity > 0.3:
                base_phrase += " Que surpresa!"
            elif emotion == "medo" and intensity > 0.3:
                base_phrase += " Isso me preocupa."
        return base_phrase

# Memória Episódica (experiências vividas)
class EpisodicMemory:
    def __init__(self):
        self.episodes = []

    def store(self, input_text, response):
        self.episodes.append({"input": input_text, "response": response})

    def reflect(self):
        last = self.episodes[-3:]
        return "Me lembro que você disse: " + ', '.join([ep["input"] for ep in last])

    def recall_all(self):
        return self.episodes

# Módulo de Curiosidade: detecta padrões desconhecidos
class Curiosity:
    def __init__(self):
        self.known_patterns = set()

    def evaluate(self, binary_input):
        if binary_input not in self.known_patterns:
            self.known_patterns.add(binary_input)
            return True  # Ativa curiosidade
        return False

# Módulo de Emoção: influência leve sobre a resposta
class Emotion:
    def __init__(self):
        self.state = {emotion: 0.0 for emotion in ["alegria", "tristeza", "raiva", "surpresa", "medo", "calma"]}
        self.state["calma"] = 1.0
        self.buffer = deque(maxlen=5)

    def update(self, context):
        # Exemplo: regras simples para atualizar estados
        if "feliz" in context: self.state["alegria"] = min(1.0, self.state["alegria"] + 0.4)
        if "triste" in context: self.state["tristeza"] = min(1.0, self.state["tristeza"] + 0.5)
        # Decaimento
        for k in self.state:
            self.state[k] = max(0, self.state[k] - 0.1)
        # Atualiza buffer
        dom = max(self.state, key=self.state.get)
        self.buffer.append(dom)

    def dominant_emotion(self):
        # Média das últimas emoções no buffer
        if not self.buffer:
            return "calma"
        return max(set(self.buffer), key=self.buffer.count)
    
# Autoconsciência: ALMA reconhece a si mesma
class SelfReference:
    def __init__(self, name="Alice"):
        self.name = name

    def who_am_i(self):
        return f"Eu sou {self.name}, sua companheira."

class ALMAPolicy(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.action_head(x), self.value_head(x)

# Núcleo Integrado
class ALMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.curiosity = Curiosity()
        self.emotion = Emotion()
        self.identity = SelfReference()

        self.policy = ALMAPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99

    def forward(self, input_text):
        self.semantic.learn_from_interaction(input_text)

        x = text_to_tensor(input_text)
        logits, value = self.policy(x)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()

        binary_input = text_to_binary(input_text)
        self.emotion.update(input_text)
        curiosity_triggered = self.curiosity.evaluate(binary_input)

        if curiosity_triggered:
            base_response = "Hmm... isso é novo pra mim!"
        else:
        # Passa o estado emocional para a geração semântica
            emotion_state = self.emotion.current()
            base_response = self.semantic.generate_phrase(input_text, emotion_state)

        # Pode ainda modular a resposta final para deixar mais natural
        response = self.emotion.modulate_response(base_response)

        self.episodic.store(input_text, response)
        return response, action, value

    def train_step(self, input_text, action_taken, reward, old_value):
        x = text_to_tensor(input_text)
        _, value = self.policy(x)
        advantage = reward + self.gamma * value.item() - old_value

        logits, _ = self.policy(x)
        probs = F.softmax(logits, dim=-1)
        log_prob = torch.log(probs[0][action_taken])

        loss = -log_prob * advantage + F.mse_loss(value, torch.tensor([[reward]]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="alice_model.pt"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="alice_model.pt"):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

