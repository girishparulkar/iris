import json
import os
import requests
from flask import Flask, request, render_template_string

app = Flask(__name__)

ENDPOINT_URL = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/iris-demo-endpoint/v2/models/00zs-5ozn-ebe8-4nr5/infer"
API_KEY = "TWpnM1pqZ3lNMkV0WWpNNVppMDBNemt5TFdFNVkyVXRPV1ExTldGbVlqYzROVGcxOjpaamMxWmprMU5tTXRaR05tTXkwME1ERmtMV0UyTXpZdE1ESTBNalF6TURKaU5UUmw="

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
