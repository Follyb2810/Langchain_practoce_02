```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/success")
def success():
    return jsonify(message="Everything OK"), 200  # status 200 OK

@app.route("/notfound")
def not_found():
    return jsonify(error="Resource not found"), 404  # status 404 Not Found

if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, jsonify, make_response

app = Flask(__name__)

@app.route("/unauthorized")
def unauthorized():
    response = make_response(jsonify(error="Unauthorized"), 401)
    response.headers["X-Custom-Header"] = "Denied"
    return response


from flask import Flask, abort

app = Flask(__name__)

@app.route("/forbidden")
def forbidden():
    abort(403, description="You don’t have access to this resource.")


from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/user/<int:user_id>")
def get_user(user_id):
    user = {"id": user_id, "name": "Alice"}
    return jsonify(user), 200   # data + status

from flask import Flask, jsonify, make_response

app = Flask(__name__)

@app.route("/login")
def login():
    data = {"message": "Login failed"}
    response = make_response(jsonify(data), 401)  # data + status
    response.headers["X-Reason"] = "Invalid credentials"
    return response

from flask import Flask, jsonify, abort

app = Flask(__name__)

@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

@app.route("/item/<int:item_id>")
def get_item(item_id):
    items = {1: "Laptop"}
    if item_id not in items:
        abort(404, description="Item not found")
    return jsonify(id=item_id, name=items[item_id]), 200

@app.route("/register", methods=["POST"])
def register():
    user = {"id": 123, "name": "Folly"}
    return jsonify(user=user, message="User created"), 201  # 201 Created



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/protected")
def protected():
    auth = request.authorization
    if auth and auth.username == "admin" and auth.password == "secret":
        return jsonify(message="Welcome, admin"), 200
    return jsonify(error="Unauthorized"), 401


from flask import Flask, request, jsonify

app = Flask(__name__)

TOKENS = {"abc123": "folly"}  # fake token store

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if data.get("username") == "folly" and data.get("password") == "mypassword":
        return jsonify(token="abc123")  # normally you’d generate this dynamically
    return jsonify(error="Invalid credentials"), 401

@app.route("/dashboard")
def dashboard():
    token = request.headers.get("Authorization")
    if token == "Bearer abc123":
        return jsonify(message="Welcome to your dashboard, folly")
    return jsonify(error="Unauthorized"), 401


pip install flask-jwt-extended
from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "super-secret-key"  # use env var in real apps

jwt = JWTManager(app)

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    if username == "folly" and password == "mypassword":
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    return jsonify(error="Bad credentials"), 401

@app.route("/profile")
@jwt_required()
def profile():
    current_user = get_jwt_identity()
    return jsonify(message=f"Hello {current_user}, welcome back!")

from flask import Flask, session, request, jsonify

app = Flask(__name__)
app.secret_key = "secret"

@app.route("/login", methods=["POST"])
def login():
    if request.json.get("username") == "folly":
        session["user"] = request.json["username"]
        return jsonify(message="Logged in")
    return jsonify(error="Bad credentials"), 401

@app.route("/me")
def me():
    if "user" in session:
        return jsonify(user=session["user"])
    return jsonify(error="Unauthorized"), 401


```