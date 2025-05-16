import requests

def emit_message(msg: str):
    print("[HTTP Emit] Send message to Flask:", msg)
    try:
        requests.post("http://localhost:5000/message", json={"msg": msg})
    except Exception as e:
        print(f"[HTTP Emit] Error in sending message: {e}")