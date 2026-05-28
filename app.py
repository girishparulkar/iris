import json
import os
import requests
from flask import Flask, request, render_template_string

app = Flask(__name__)

ENDPOINT_URL = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/iris-demo-endpoint/v2/models/00zs-5ozn-ebe8-4nr5/infer"
API_KEY = "eyJraWQiOiIzYzhlNzA3OTEyZmI0NTA1ODE3NzE3YzMyOTU4MmQwMTFjYjlmNTAwIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnaXJpc2hwIiwiYXVkIjoiaHR0cHM6Ly9kZS55bGN1LWF0bWkuY2xvdWRlcmEuc2l0ZSIsImlzcyI6Imh0dHBzOi8vY29uc29sZWF1dGguY2RwLmNsb3VkZXJhLmNvbS84YTFlMTVjZC0wNGMyLTQ4YWEtOGYzNS1iNGE4YzExOTk3ZDMiLCJncm91cHMiOiJjZHBfZGVtb3Nfd29ya2Vyc193dyBjZHBfZGVtby1hd3MtcHJpbSBfY19kZl9kZXZlbG9wXzkxMTQ2M2MgX2NfZGZfdmlld182ZjU5ZTlmMyBfY19kZl9hZG1pbmlzdGVyXzkxMTQ2M2MgX2NfZGZfdmlld185MTE0NjNjIF9jX21sX2J1c2luZXNzX3VzZXJzXzZlZTBkYjkxIF9jX2RmX3B1Ymxpc2hfOTExNDYzYyBfY19kZl92aWV3XzkxMTQ2M2MwIF9jX2Vudl9hc3NpZ25lZXNfOTExNDYzYyBfY19yYW5nZXJfYWRtaW5zXzkwNmIwYmEgX2NfbWxfdXNlcnNfNmY1OWU5ZjMgX2NfbWxfdXNlcnNfNGQ4M2FkN2YgX2NfZW52X2Fzc2lnbmVlc185MDZiMGJhIF9jX2RmX2RldmVsb3BfNmY1OWU5ZjMgX2NfZGZfYWRtaW5pc3Rlcl82ZjU5ZTlmMyBfY19kZl9wdWJsaXNoXzZmNTllOWYzIF9jX21sX3VzZXJzXzZlZTBkYjkxIF9jX21sX2J1c2luZXNzX3VzZXJzXzZmNTllOWYzIF9jX21sX2J1c2luZXNzX3VzZXJzXzkxMTQ2M2MgX2NfZW52X2Fzc2lnbmVlc182ZjU5ZTlmMyBfY19yYW5nZXJfYWRtaW5zXzZmNTllOWYzIF9jX3Jhbmdlcl9hZG1pbnNfOTExNDYzYyBfY19kZV91c2Vyc185MTE0NjNjIF9jX21sX3VzZXJzXzkxMTQ2M2MgX2NfZGZfcHJvamVjdF9tZW1iZXJfNDBkZmU1NjggX2Nfa25veF9hZG1pbnNfOTExNDYzYyBfY19kZl92aWV3XzZmNTllOWYzMCBfY19kZl9wcm9qZWN0X21lbWJlcl81NzVmODRmNyBfY19kZV91c2Vyc182ZjU5ZTlmMyIsImV4cCI6MTc3OTk5NDk2NSwidHlwZSI6InVzZXIiLCJnaXZlbl9uYW1lIjoiR2lyaXNoIiwiaWF0IjoxNzc5OTkxMzY1LCJmYW1pbHlfbmFtZSI6IlBhcnVsa2FyIiwiZW1haWwiOiJnaXJpc2hwQGNsb3VkZXJhLmNvbSJ9.fqVmDPPpWGopkMwKj69iIqUQzi4QzuAV5VK6-8TpFlV_eYBWXUevSmYcCkA7Un_DHTZz_4FYs48T4iPugCaKWzM8f_v996nDl8BhLvXaTwP5Xtp0bkKN2KNjCmTGBR2ZjAppAx06xzoVUpf_4UGjLD-PpVJO6WAFvY_fbP2IkpK-xi3tQKcNrgn61DBElByUebjsOI_e7-GtJSVc_Ao0clh0H17I145hFfF_DGaEf_80GUfR7yqioYq7a2oikWUYnYJggJihcHfDgj1BO9RActhmDlZQQ3zrb-KrNoUsRa_f0DXhiKLYhPTTcY3SplMDBUJ1GHyr23xhwBq-U5RgHg"

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

LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

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

            try:
                result = r.json()
                ctx["response_text"] = json.dumps(result, indent=2)

                if (
                    "outputs" in result and
                    len(result["outputs"]) > 0 and
                    "data" in result["outputs"][0] and
                    len(result["outputs"][0]["data"]) > 0
                ):
                    pred_value = int(result["outputs"][0]["data"][0])
                    ctx["predicted_label"] = LABELS.get(pred_value, f"Unknown ({pred_value})")

            except Exception:
                ctx["response_text"] = r.text if r.text else "<empty response>"

        except Exception as e:
            ctx["error"] = str(e)

    return render_template_string(HTML, **ctx)

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
