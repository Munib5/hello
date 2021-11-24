from pathlib import Path
from ser.constants import RESULTS_DIR

def vis(pred, confidence, pixels):
    print(generate_ascii_art(pixels))
    print(f"The prediction is a {pred}")
    print(f"The confidence is {confidence*100}%")

def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)

def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "

def table_runs(run_path):
    path = run_path.parent.parent
    i = 0
    run_names = []
    experiment_names = []
    print('{:<4}'.format('Experiment'), '{:<4}'.format('\t Runs'), end='\n')
    print('-----------------------------------------------------------------')
    for p in path.glob("*/"):
        # because path is object not string
        experiment_names.append(str(p.name))
    for e in experiment_names:
        run_path_internal = Path(RESULTS_DIR/f"{e}/")
        for p in run_path_internal.glob("*/"):
            # because path is object not string
            print('{:<4}'.format(e), '{:<4}'.format(str(p.name)), end='\n')
    