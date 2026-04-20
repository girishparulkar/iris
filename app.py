import json
import os
import requests
from flask import Flask, request, render_template_string

app = Flask(__name__)

ENDPOINT_URL = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/iris-demo-endpoint/v2/models/00zs-5ozn-ebe8-4nr5/infer"

HTML = """
<!doctype html>
<html>
  <head>
    <title>CAI Demo - Iris Prediction</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; max-width: 720px; }
      input { margin: 6px 0; padding: 6px; width: 220px; }
      button { margin-top: 12px; padding: 8px 14px; }
      .box { margin-top: 20px; padding: 12px; background: #f4f4f4; border-radius: 6px; white-space: pre-wrap; }
      .err { color: #b00020; }
    </style>
  </head>
  <body>
    <h1>CAI Demo - Iris Prediction</h1>

    <form method="post">
      <div>Sepal length<br><input name="sepal_length" value="{{ sepal_length }}"></div>
      <div>Sepal width<br><input name="sepal_width" value="{{ sepal_width }}"></div>
      <div>Petal length<br><input name="petal_length" value="{{ petal_length }}"></div>
      <div>Petal width<br><input name="petal_width" value="{{ petal_width }}"></div>
      <button type="submit">Predict</button>
    </form>

    {% if status is not none %}
      <div class="box">
        <b>Status:</b> {{ status }}

{% if response_text %}
{{ response_text }}
{% endif %}
      </div>
    {% endif %}

    {% if error %}
      <div class="box err"><b>Error:</b> {{ error }}</div>
    {% endif %}
  </body>
</html>
"""

def get_api_key() -> str:
    jwt_path = "/tmp/jwt"
    if os.path.exists(jwt_path):
        with open(jwt_path, "r") as f:
            return json.load(f)["access_token"]
    return os.environ.get("IRIS_AUTH_TOKEN", "")

@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
        "status": None,
        "response_text": None,
        "error": None,
    }

    if request.method == "POST":
        try:
            ctx["sepal_length"] = request.form.get("sepal_length", "5.1")
            ctx["sepal_width"] = request.form.get("sepal_width", "3.5")
            ctx["petal_length"] = request.form.get("petal_length", "1.4")
            ctx["petal_width"] = request.form.get("petal_width", "0.2")

            payload = {
                "inputs": [[
                    float(ctx["sepal_length"]),
                    float(ctx["sepal_width"]),
                    float(ctx["petal_length"]),
                    float(ctx["petal_width"]),
                ]]
            }

            api_key = get_api_key()
            headers = {"Content-Type": "application/json"}

            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            r = requests.post(
                ENDPOINT_URL,
                json=payload,
                headers=headers,
                timeout=30
            )

            ctx["status"] = r.status_code
            try:
                ctx["response_text"] = json.dumps(r.json(), indent=2)
            except Exception:
                ctx["response_text"] = r.text

        except Exception as e:
            ctx["error"] = str(e)

    return render_template_string(HTML, **ctx)

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
