import click
from loguru import logger

from wtb.classification.bird_classifier import BirdClassifier


@click.command()
@click.option("--image_path", type=click.Path(exists=True))
@click.option("--model_path", type=click.Path(exists=True))
def predict(image_path: str, model_path: str):
    model = BirdClassifier.load_classifier(model_path)
    prediction = model.predict(image_path)

    logger.info(prediction)
    logger.info(model.get_human_prediction(prediction))


if __name__ == "__main__":
    predict()
