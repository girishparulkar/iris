import json
import os
import requests
from flask import Flask, request, render_template_string

app = Flask(__name__)

ENDPOINT_URL = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/iris-demo-endpoint/v2/models/00zs-5ozn-ebe8-4nr5/infer"
API_KEY = "eyJraWQiOiIzYzhlNzA3OTEyZmI0NTA1ODE3NzE3YzMyOTU4MmQwMTFjYjlmNTAwIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnaXJpc2hwIiwiYXVkIjoiaHR0cHM6Ly9kZS55bGN1LWF0bWkuY2xvdWRlcmEuc2l0ZSIsImlzcyI6Imh0dHBzOi8vY29uc29sZWF1dGguY2RwLmNsb3VkZXJhLmNvbS84YTFlMTVjZC0wNGMyLTQ4YWEtOGYzNS1iNGE4YzExOTk3ZDMiLCJncm91cHMiOiJjZHBfZGVtb3Nfd29ya2Vyc193dyBfY19kZl9hZG1pbmlzdGVyXzZmNTllOWYzIF9jX2RmX3B1Ymxpc2hfNmY1OWU5ZjMgX2NfZGZfZGV2ZWxvcF85MTE0NjNjIF9jX2RmX3ZpZXdfNmY1OWU5ZjMgX2NfbWxfYnVzaW5lc3NfdXNlcnNfNmY1OWU5ZjMgX2NfbWxfYnVzaW5lc3NfdXNlcnNfOTExNDYzYyBfY19kZl9hZG1pbmlzdGVyXzkxMTQ2M2MgX2NfZW52X2Fzc2lnbmVlc182ZjU5ZTlmMyBfY19yYW5nZXJfYWRtaW5zXzZmNTllOWYzIF9jX3Jhbmdlcl9hZG1pbnNfOTExNDYzYyBfY19kZV91c2Vyc185MTE0NjNjIF9jX2RmX3ZpZXdfOTExNDYzYyBfY19tbF91c2Vyc185MTE0NjNjIF9jX2RmX3B1Ymxpc2hfOTExNDYzYyBfY19kZl92aWV3XzkxMTQ2M2MwIF9jX2Vudl9hc3NpZ25lZXNfOTExNDYzYyBfY19tbF91c2Vyc182ZjU5ZTlmMyBfY19kZl92aWV3XzZmNTllOWYzMCBfY19tbF91c2Vyc180ZDgzYWQ3ZiBfY19kZl9kZXZlbG9wXzZmNTllOWYzIF9jX2RlX3VzZXJzXzZmNTllOWYzIiwiZXhwIjoxNzc2NzE0NjM4LCJ0eXBlIjoidXNlciIsImdpdmVuX25hbWUiOiJHaXJpc2giLCJpYXQiOjE3NzY3MTEwMzgsImZhbWlseV9uYW1lIjoiUGFydWxrYXIiLCJlbWFpbCI6ImdpcmlzaHBAY2xvdWRlcmEuY29tIn0.WQe65aZx3mhxMOwT80fQxADXVDT3n0Gn-dcfScndj9v_GTZ0QzEHddPMalvAoeK7gyu2WhfD1lapdbCeUeGkSm_9Wx3AAwOBu7SVrE7lye7sgVdE1vFLAw-x8H7w1BYasvYCFKydiq7zigbWwdl5lmMVqHrRecwUd_63LMdsYGUV6HA5r9k05OIMATaJA828jQI7LwsVYe7Q_gHu3HWlhE8BtEwsFUnrZ_BCEUBVVpwfDlPx9bJm_4ehY9CsjmqF6oNc1aMZm-mvfRvLrfrwCMLbiclb_mvhgenhtibk624s0jlyljef7jmC6hBzf18aClOBn5G90SVjHQsJ2l5cdg"

LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

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
      .pred { font-size: 20px; font-weight: bold; margin-top: 12px; color: #0b6b2d; }
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

    {% if predicted_label %}
      <div class="pred">Predicted class: {{ predicted_label }}</div>
    {% endif %}

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
            result = r.json()
            ctx["response_text"] = json.dumps(result, indent=2)

            try:
                pred_value = int(result["outputs"][0]["data"][0])
                ctx["predicted_label"] = LABELS.get(pred_value, f"Unknown ({pred_value})")
            except Exception:
                pass

        except Exception as e:
            ctx["error"] = str(e)

    return render_template_string(HTML, **ctx)

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
