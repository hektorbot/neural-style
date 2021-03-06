import os
import sys
import uuid
from flask import Flask, send_file, request, Response
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import requests

from neural_style import main as evaluate

ALLOWED_EXTENSIONS = set(["jpg", "jpeg", "png"])
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    return Response("OK", status=200)


@app.route("/", methods=["POST"])
def style_transfer():
    print("Style transfer initializing...")
    input_file = request.files.get("input")
    style_file = request.files.get("style")
    job_id = request.values.get("job_id")
    cb_url = request.values.get("cb_url")

    iterations = str(request.values.get("iterations", 1000))
    style_layer_weight_exp = str(request.values.get("style_layer_weight_exp", 1.0))
    content_weight_blend = str(request.values.get("content_weight_blend", 1.0))
    pooling = str(request.values.get("pooling", "avg"))
    preserve_colors = str(request.values.get("preserve_color", False))

    if not input_file or not style_file:
        return BadRequest("File not present in request")
    if not job_id:
        return BadRequest("Job ID not present in request")
    if not cb_url:
        return BadRequest("Callback URL not present in request")

    input_filename = secure_filename(input_file.filename)
    style_filename = secure_filename(style_file.filename)

    if input_filename == "" or style_filename == "":
        return BadRequest("File name is not present in request")
    if not allowed_file(input_filename) or not allowed_file(style_filename):
        return BadRequest("Invalid file type")

    input_filepath = os.path.join("./images/", input_filename)
    style_filepath = os.path.join("./images/", style_filename)
    output_filepath = os.path.join("/output/", job_id + ".jpg")
    input_file.save(input_filepath)
    style_file.save(style_filepath)

    sys.argv.extend(["--content", input_filepath])
    sys.argv.extend(["--styles", style_filepath])
    sys.argv.extend(["--output", output_filepath])
    sys.argv.extend(["--network", "/vgg/imagenet-vgg-verydeep-19.mat"])

    print("--iterations: {}".format(iterations))
    print("--style-layer-weight-exp: {}".format(style_layer_weight_exp))
    print("--content-weight-blend: {}".format(content_weight_blend))
    print("--pooling: {}".format(pooling))
    print("--preserve-colors: {}".format(preserve_colors))

    sys.argv.extend(["--iterations", iterations])
    sys.argv.extend(["--style-layer-weight-exp", style_layer_weight_exp])
    sys.argv.extend(["--content-weight-blend", content_weight_blend])
    sys.argv.extend(["--pooling", pooling])
    if preserve_colors is "1":
        sys.argv.extend(["--preserve-colors"])

    evaluate()

    # Send result back to the app
    print("Sending output to {}".format(cb_url))
    output_file = open(output_filepath, "rb")
    files = {"file": output_file}
    requests.post(cb_url, files=files, data={"job_id": job_id})
    return "OK"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(host="0.0.0.0")

