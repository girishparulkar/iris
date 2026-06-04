import json
import os
import requests
from flask import Flask, request, render_template_string

app = Flask(__name__)

ENDPOINT_URL = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/inf-iris-pkl-dcb/v2/models/yes4-g9wy-35ln-e1t7/infer"
API_KEY = "eyJraWQiOiIzYzhlNzA3OTEyZmI0NTA1ODE3NzE3YzMyOTU4MmQwMTFjYjlmNTAwIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnaXJpc2hwIiwiYXVkIjoiaHR0cHM6Ly9kZS55bGN1LWF0bWkuY2xvdWRlcmEuc2l0ZSIsImlzcyI6Imh0dHBzOi8vY29uc29sZWF1dGguY2RwLmNsb3VkZXJhLmNvbS84YTFlMTVjZC0wNGMyLTQ4YWEtOGYzNS1iNGE4YzExOTk3ZDMiLCJncm91cHMiOiJjZHBfZGVtb3Nfd29ya2Vyc193dyBjZHBfZGVtby1hd3MtcHJpbSBfY19kZl9kZXZlbG9wXzkxMTQ2M2MgX2NfZGZfdmlld182ZjU5ZTlmMyBfY19kZl9hZG1pbmlzdGVyXzkxMTQ2M2MgX2NfZGZfdmlld185MTE0NjNjIF9jX21sX2J1c2luZXNzX3VzZXJzXzZlZTBkYjkxIF9jX2RmX3B1Ymxpc2hfOTExNDYzYyBfY19kZl92aWV3XzkxMTQ2M2MwIF9jX2Vudl9hc3NpZ25lZXNfOTExNDYzYyBfY19yYW5nZXJfYWRtaW5zXzkwNmIwYmEgX2NfbWxfdXNlcnNfNmY1OWU5ZjMgX2NfbWxfdXNlcnNfNGQ4M2FkN2YgX2NfZW52X2Fzc2lnbmVlc185MDZiMGJhIF9jX2RmX2RldmVsb3BfNmY1OWU5ZjMgX2NfZGZfYWRtaW5pc3Rlcl82ZjU5ZTlmMyBfY19kZl9wdWJsaXNoXzZmNTllOWYzIF9jX21sX3VzZXJzXzZlZTBkYjkxIF9jX21sX2J1c2luZXNzX3VzZXJzXzZmNTllOWYzIF9jX21sX2J1c2luZXNzX3VzZXJzXzkxMTQ2M2MgX2NfZW52X2Fzc2lnbmVlc182ZjU5ZTlmMyBfY19yYW5nZXJfYWRtaW5zXzZmNTllOWYzIF9jX3Jhbmdlcl9hZG1pbnNfOTExNDYzYyBfY19kZV91c2Vyc185MTE0NjNjIF9jX21sX3VzZXJzXzkxMTQ2M2MgX2NfZGZfcHJvamVjdF9tZW1iZXJfNDBkZmU1NjggX2Nfa25veF9hZG1pbnNfOTExNDYzYyBfY19kZl92aWV3XzZmNTllOWYzMCBfY19kZl9wcm9qZWN0X21lbWJlcl81NzVmODRmNyBfY19kZV91c2Vyc182ZjU5ZTlmMyIsImV4cCI6MTc4MDU4MDMwMCwidHlwZSI6InVzZXIiLCJnaXZlbl9uYW1lIjoiR2lyaXNoIiwiaWF0IjoxNzgwNTc2NzAwLCJmYW1pbHlfbmFtZSI6IlBhcnVsa2FyIiwiZW1haWwiOiJnaXJpc2hwQGNsb3VkZXJhLmNvbSJ9.wdIwgSBsQamKyeO8zX-4RVREt7_VAmekjgc6I7YRNTJaqCw5z-6XgXImwkEeQVCnTLjYNkn7Z3BkD3iFTnKh2xo9hR8OcAO-tgZMcOToABCNJsCgBlacxrFNEs8lsxhk0tpehRzMp1Z55-s8evu2h39qwUb-xb3I2c6mB472Qek-0Uq9PL9lbb1xwifIW6ehgV6UOwXOl2N98_ljsKV4Z2jKPZtV9Sd5XRua_J0gCLwFTA6uSqYx0jPFjhj-IHONbHxTbDcurPoB09WsWFcBP1X2Kxjfl1hCPrbZ67p5aXbOmyNvCng9iZONH_f_zaV-XSSQSOsFkrkQMpU8K_i1AA"

LABELS = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica",
}

HTML = """
<!doctype html>
<html>
  <head>
    <title>CAI Demo - Iris Prediction</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; max-width: 760px; }
      h1 { margin-bottom: 30px; }
      .field { margin-bottom: 14px; }
      label { display: block; margin-bottom: 4px; font-weight: 600; }
      input { padding: 8px; width: 240px; }
      button { margin-top: 10px; padding: 10px 16px; cursor: pointer; }
      .result {
        margin-top: 24px;
        padding: 16px;
        background: #f4f4f4;
        border-radius: 8px;
      }
      .pred {
        font-size: 24px;
        font-weight: 700;
        color: #0b6b2d;
        margin-bottom: 10px;
      }
      .sub {
        color: #444;
        margin-bottom: 8px;
      }
      .err {
        margin-top: 24px;
        padding: 16px;
        background: #fff1f1;
        color: #b00020;
        border-radius: 8px;
      }
      details { margin-top: 12px; }
      pre { white-space: pre-wrap; word-break: break-word; }
    </style>
  </head>
  <body>
    <h1>CAI Demo - Iris Prediction</h1>

    <form method="post">
      <div class="field">
        <label>Sepal length</label>
        <input name="sepal_length" value="{{ sepal_length }}">
      </div>
      <div class="field">
        <label>Sepal width</label>
        <input name="sepal_width" value="{{ sepal_width }}">
      </div>
      <div class="field">
        <label>Petal length</label>
        <input name="petal_length" value="{{ petal_length }}">
      </div>
      <div class="field">
        <label>Petal width</label>
        <input name="petal_width" value="{{ petal_width }}">
      </div>
      <button type="submit">Predict</button>
    </form>

    {% if predicted_label %}
      <div class="result">
        <div class="pred">Predicted class: {{ predicted_label }}</div>
        <div class="sub">Numeric prediction: {{ predicted_value }}</div>

        <details>
          <summary>Show raw response</summary>
          <pre>{{ response_text }}</pre>
        </details>
      </div>
    {% endif %}

    {% if status is not none and not predicted_label %}
      <div class="result">
        <div><b>Status:</b> {{ status }}</div>
        <pre>{{ response_text }}</pre>
      </div>
    {% endif %}

    {% if error %}
      <div class="err">
        <b>Error:</b> {{ error }}
      </div>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
        "status": None,
        "response_text": None,
        "predicted_label": None,
        "predicted_value": None,
        "error": None,
    }

    if request.method == "POST":
        try:
            ctx["sepal_length"] = request.form.get("sepal_length", "5.1")
            ctx["sepal_width"] = request.form.get("sepal_width", "3.5")
            ctx["petal_length"] = request.form.get("petal_length", "1.4")
            ctx["petal_width"] = request.form.get("petal_width", "0.2")

            values = [
                float(ctx["sepal_length"]),
                float(ctx["sepal_width"]),
                float(ctx["petal_length"]),
                float(ctx["petal_width"]),
            ]

            payload = {
                "inputs": [
                    {
                        "name": "INPUT__0",
                        "shape": [1, 4],
                        "datatype": "FP32",
                        "data": [values]
                    }
                ]
            }

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }

            r = requests.post(
                ENDPOINT_URL,
                json=payload,
                headers=headers,
                timeout=30
            )

            ctx["status"] = r.status_code

            try:
                result = r.json()
                ctx["response_text"] = json.dumps(result, indent=2)

                pred_value = int(result["outputs"][0]["data"][0])
                ctx["predicted_value"] = pred_value
                ctx["predicted_label"] = LABELS.get(pred_value, f"Unknown ({pred_value})")
            except Exception:
                ctx["response_text"] = r.text if r.text else "<empty response>"

        except Exception as e:
            ctx["error"] = str(e)

    return render_template_string(HTML, **ctx)

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
