import os

import click
from classifier import MODELS_DIR, Classifier
from dataset import SHAPES_DIR, Dataset


@click.group()
def cli() -> None:
    if not os.path.exists(SHAPES_DIR):
        Dataset()


@cli.command(help="Trains the model for a fixed number of epochs (dataset iterations).")
@click.option("-b", "--batch", default=32, help="Number of samples per gradient update.")
@click.option("-e", "--epochs", default=32, help="Number of epochs to train the model.")
def train(batch: int, epochs: int) -> None:
    classifier = Classifier()
    classifier.load_best_results()
    classifier.fit_and_evaluate(batch, epochs)


@cli.command(help="Solves a captcha image puzzle.")
@click.argument("captcha", type=click.Path(exists=True))
def solve(captcha: str) -> None:
    classifier = Classifier(
        model_path=os.path.join(MODELS_DIR, "shapes_best.keras"),
        verbose=0
    )
    classes = classifier.predict(captcha)
    solution = ",".join([str(i) for i in range(1, len(classes))
                         if classes[0] == classes[i]])
    click.echo(f"Answer: {solution}")


cli()
