import wandb
run = wandb.init()
artifact = run.use_artifact('forl-traffic/PMLR/model-4yobs5ix:best', type='model')
artifact_dir = artifact.download()
print(f"Downloaded artifact to {artifact_dir}")
