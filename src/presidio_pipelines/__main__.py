"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Presidio Pipelines."""


if __name__ == "__main__":
    main(prog_name="presidio-pipelines")  # pragma: no cover
