import os
import traceback

import click
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from flask_restplus import Api, Resource, abort
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from loguru import logger
from requests import codes as http_codes
from werkzeug.utils import secure_filename

API_CONF = {
    "version": "1.0.0",
    "host": "localhost",
    "port": 9090,
    "url_prefix": "api-birdhouse-1"
}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
PICTURE_FOLDER = "photos"
CLASSIFICATION_MODEL = 'models/bestmodel-09-0.97.hdf5'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path):
    response = {
    }
    try:
        picture = Image.open(image_path)
        picture = picture.resize(size=(224, 224))
        picture_array = img_to_array(img=picture)
        picture_array = np.expand_dims(picture_array, axis=0)
        prediction = app.model.predict(preprocess_input(picture_array))

        logger.info(prediction)

        # with open(mapper_path, 'r') as fp:
        #     mapper = json.load(fp)

        best_proba_index = np.argmax(prediction, axis=1)[0]
        # logger.info("{} ({:.1f}%)".format(mapper.get(str(best_proba_index)), prediction[0][best_proba_index] * 100))

        response = {
            'prediction': int(best_proba_index)
        }
    except Exception:
        abort(http_codes.SERVER_ERROR, "Internal error while classifying the file")
    return response


def init_app(p_conf):
    # Load app config into Flask WSGI running instance
    r_app = Flask(__name__)
    r_app.config["API_CONF"] = p_conf

    # Authorize Cross-origin (CORS)
    r_app.config["CORS_HEADERS"] = "Auth-Token, Content-Type, User, Content-Length"
    CORS(r_app, resources={r"/*": {"origins": "*"}})

    blueprint = Blueprint("api", __name__)
    r_swagger_api = Api(
        blueprint,
        doc=f'/{p_conf["url_prefix"]}/doc/',
        title="API",
        description="Automatic bird classification",
    )
    r_app.register_blueprint(blueprint)
    r_ns = r_swagger_api.namespace(
        name=p_conf["url_prefix"], description="Api documentation"
    )

    return r_app, r_swagger_api, r_ns


app, swagger_api, ns = init_app(API_CONF)


# Access log query interceptor
@app.before_request
def access_log():
    logger.info(f"{request.method} {request.path}")


@ns.route("/", strict_slashes=False)
class Base(Resource):
    @staticmethod
    def get():
        response = {"status_code": http_codes.OK, "message": "Api birdhouse"}

        return make_reponse(response, http_codes.OK)


@ns.route("/heartbeat")
class Heart(Resource):
    @staticmethod
    def get():
        response = {"status_code": http_codes.OK, "message": "Heartbeat"}

        return make_reponse(response, http_codes.OK)


@ns.route("/supervision")
class Supervision(Resource):
    @staticmethod
    def get():
        response = None
        try:
            response = app.config["API_CONF"]
        except Exception:
            abort(http_codes.SERVER_ERROR, "Can't get the configuration")

        return _success(response)


# Doc
prediction_route_parser = swagger_api.parser()


@swagger_api.expect(prediction_route_parser)
@ns.route("/bird", endpoint="/bird")
class BirdClassification(Resource):
    @staticmethod
    def post():
        file = request.files['file']
        if file:  # and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(PICTURE_FOLDER, "uploaded_picture")
            file.save(file_path)
            response = predict(file_path)

        return _success(response)


def _success(response):
    return make_reponse(response, http_codes.OK)


def _failure(exception, http_code=http_codes.SERVER_ERROR):
    try:
        exn = traceback.format_exc(exception)
        logger.info("EXCEPTION: {}".format(exn))
    except Exception as e:
        logger.info("EXCEPTION: {}".format(exception))
        logger.info(e)

    try:
        data, code = exception.to_tuple()
        return make_reponse(data, code)
    except:
        try:
            data = exception.to_dict()
            return make_reponse(data, exception.http)
        except Exception:
            return make_reponse(None, http_code)


def make_reponse(p_object=None, status_code=http_codes.OK):
    json_response = jsonify(p_object)
    json_response.status_code = status_code
    json_response.content_type = "application/json;charset=utf-8"
    return json_response


@click.command()
@click.option("--model_path", type=click.Path(exists=True))
def run_api(model_path: str):
    app.model = load_model(model_path)
    cf_port = os.getenv("PORT")
    if cf_port is None:
        app.run(host="0.0.0.0", port=9090, threaded=False)
    else:
        app.run(host="0.0.0.0", port=int(cf_port), threaded=False)


if __name__ == "__main__":
    run_api()
