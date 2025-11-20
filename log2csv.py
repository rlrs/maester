import aimrun
import typer

def main(
    log_file: str,
    verbose: bool = False,
    aim: bool = False,
    color: str = "red",
):
    lines = open(log_file,"rt").readlines()
    step2line = {}
    step2loss = {}
    for line in lines:
        if "time/data_loading=" in line:
            step = int(line.split("Step ")[1].split(":")[0])
            step2line[step] = line.strip()
            loss = float(line.split("loss= ")[1].split()[0])
            step2loss[step] = loss
    if verbose:
        print("step,loss")
        for step, loss in step2loss.items():
            print(f"{step},{loss}")
    if aim:
        aimrun._init(repo=".", experiment=log_file, args={}, run_hash=None)
        run = aimrun.get_runs()[0]
        hash = run.hash
        print(f'    - {{hash: {hash}, color: {color}, label: "{log_file}"}}')
        for step, loss in step2loss.items():
            aimrun.track(value=loss, name="loss", step=step, context="train")
        aimrun.close()

if __name__ == "__main__":
    typer.run(main)
