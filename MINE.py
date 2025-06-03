import socket

# Supondo que o bot Mineflayer esteja escutando em localhost:3000
MINE_SOCKET_HOST = "localhost"
MINE_SOCKET_PORT = 3000

def send_minecraft_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((MINE_SOCKET_HOST, MINE_SOCKET_PORT))
            s.sendall(command.encode('utf-8'))
    except Exception as e:
        print(f"[ERRO] Falha ao enviar comando pro Mine: {e}")
