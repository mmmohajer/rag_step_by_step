BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"


def step(number, title):
    print(f"\n{CYAN}{BOLD}Step {number}{RESET}  {title}", flush=True)


def done(detail):
    print(f"        {GREEN}done{RESET}  {DIM}{detail}{RESET}", flush=True)
