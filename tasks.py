from invoke import task


@task(default=True)
def run(ctx):
    ctx.run("python -m ai_orb.main")

