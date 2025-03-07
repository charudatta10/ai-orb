from invoke import task


@task(default=True)
def run(ctx):
    ctx.run("python agent/agent.py")
    ctx.run("python agent/runner.py")
