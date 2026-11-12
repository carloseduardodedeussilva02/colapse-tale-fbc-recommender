"""Servidor do Colapse!Tale para jogar na mesma rede Wi-Fi.

Ele só cria salas e repassa mensagens entre os jogadores. As regras rodam no navegador
de quem criou a sala (o anfitrião), então basta uma pessoa deixar a página aberta.

Uso:  pip install flask flask-sock   e depois   python app.py
"""
import json
import os
import random
import socket
import string
import threading

from flask import Flask, render_template, send_from_directory
from flask_sock import Sock

BASE = os.path.dirname(os.path.abspath(__file__))
PORT = 5000

app = Flask(__name__)
sock = Sock(app)
rooms = {}
lock = threading.Lock()


def enviar(ws, obj):
    try:
        with lock:
            ws.send(json.dumps(obj))
    except Exception:
        pass

@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE, "templates"), "index.html")

@sock.route("/ws")
def ws_handler(ws):
    sala, vaga = None, None
    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break
            m = json.loads(raw)
            t = m.get("t")
            if t == "create" and sala is None:
                code = "".join(random.choices(string.ascii_uppercase, k=4))
                while code in rooms:
                    code = "".join(random.choices(string.ascii_uppercase, k=4))
                sala = rooms[code] = {"code": code, "conn": {0: ws},
                                      "names": {0: str(m.get("name", "Host"))[:12]}}
                vaga = 0
                enviar(ws, {"t": "joined", "code": code, "seat": 0, "names": sala["names"]})
            elif t == "join" and sala is None:
                r = rooms.get(str(m.get("code", "")).strip().upper())
                if not r:
                    enviar(ws, {"t": "err", "msg": "Sala não encontrada."})
                    continue
                livres = [s for s in (1, 2, 3) if s not in r["conn"]]
                if not livres:
                    enviar(ws, {"t": "err", "msg": "Sala cheia."})
                    continue
                sala, vaga = r, livres[0]
                sala["conn"][vaga] = ws
                sala["names"][vaga] = str(m.get("name", "Jogador"))[:12]
                enviar(ws, {"t": "joined", "code": sala["code"], "seat": vaga, "names": sala["names"]})
                for s, w in list(sala["conn"].items()):
                    if s != vaga:
                        enviar(w, {"t": "lobby", "names": sala["names"]})
            elif sala and t == "toHost":
                enviar(sala["conn"][0], {"t": "act", "seat": vaga, "m": m.get("m")})
            elif sala and vaga == 0 and t == "toSeat":
                w = sala["conn"].get(m.get("seat"))
                if w:
                    enviar(w, m.get("m"))
            elif sala and vaga == 0 and t == "toAll":
                for s, w in list(sala["conn"].items()):
                    if s != 0:
                        enviar(w, m.get("m"))
    except Exception:
        pass
    finally:
        if sala is not None and vaga is not None:
            sala["conn"].pop(vaga, None)
            if vaga == 0:
                for w in list(sala["conn"].values()):
                    enviar(w, {"t": "closed"})
                rooms.pop(sala["code"], None)
            else:
                sala["names"].pop(vaga, None)
                for w in list(sala["conn"].values()):
                    enviar(w, {"t": "lobby", "names": sala["names"]})
                if 0 in sala["conn"]:
                    enviar(sala["conn"][0], {"t": "left", "seat": vaga})


def ip_local():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
