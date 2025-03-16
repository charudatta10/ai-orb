from invoke import task


@task(default=True)
def run(ctx):
    ctx.run("python agent/agent.py")

