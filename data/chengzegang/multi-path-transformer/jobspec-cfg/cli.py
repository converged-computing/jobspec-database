import typer
import yaml
import scripts.train_llm

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(config_path: str):
    config = yaml.full_load(open(config_path))
    scripts.train_llm.train(**config)


if __name__ == "__main__":
    app()
