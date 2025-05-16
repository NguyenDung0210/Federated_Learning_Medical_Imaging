import requests

def emit_message(msg: str):
    print("[HTTP Emit] Gửi message về Flask:", msg)
    try:
        requests.post("http://localhost:5000/message", json={"msg": msg})
    except Exception as e:
        print(f"[HTTP Emit] Lỗi khi gửi message: {e}")