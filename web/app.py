from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("chat_message")
def handle_chat(msg):
    sender_id = request.sid
    print(f"[Chat] {sender_id} send:", msg)
    # Gửi lại cho tất cả clients kèm theo ID của người gửi
    socketio.emit("chat_message", {"msg": msg, "sender_id": sender_id})

@socketio.on("fl_message")
def handle_fl_message(msg):
    print("[FL] Got message:", msg)
    socketio.emit("fl_message", msg)

@app.route("/message", methods=["POST"])
def receive_message():
    data = request.get_json()
    msg = data.get("msg", "")
    print("[HTTP] Got from Flower:", msg)
    socketio.emit("fl_message", msg)
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
