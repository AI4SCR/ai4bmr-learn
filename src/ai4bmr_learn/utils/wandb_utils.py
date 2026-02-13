import wandb

def get_runs(filters: dict, project_name: str):
    api = wandb.Api()
    runs = api.runs(
        path=f"chuv/{project_name}",
        filters=filters
    )
    return runs


def get_run(run_id: str, project_name: str):
    api = wandb.Api()
    run = api.run(f"chuv/{project_name}/{run_id}")
    return run