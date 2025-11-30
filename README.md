# curve-fitting
"Implement linear regression functions in README
Added detailed implementation of linear regression fitting, prediction, and R-squared calculation."

from typing import Sequence, Tuple, List
import argparse
import sys


def parse_sequence(s: str) -> List[float]:
    
  if s is None:
        return [] 
    # allow commas or whitespace as separators
    s = s.strip()
    if not s:
        return []
    # replace commas with spaces then split
    parts = s.replace(",", " ").split()
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"could not parse numbers from '{s}': {e}")


def linear_fit(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        raise ValueError("need at least two data points to fit a line")
    x_list = list(x)
    y_list = list(y)
    x_mean = sum(x_list) / n
    y_mean = sum(y_list) / n
    num = 0.0
    den = 0.0
    for xi, yi in zip(x_list, y_list):
        dx = xi - x_mean
        num += dx * (yi - y_mean)
        den += dx * dx
    if den == 0.0:
        raise ValueError("zero variance in x; cannot fit a unique line")
    a = num / den
    b = y_mean - a * x_mean
    return a, b


def predict(a: float, b: float, xs: Sequence[float]) -> List[float]:
    return [a * xi + b for xi in xs]


def r_squared(x: Sequence[float], y: Sequence[float], a: float, b: float) -> float:
    y_list = list(y)
    n = len(y_list)
    if n == 0:
        return 0.0
    y_mean = sum(y_list) / n
    ss_res = 0.0
    ss_tot = 0.0
    for xi, yi in zip(x, y_list):
        y_pred = a * xi + b
        ss_res += (yi - y_pred) ** 2
        ss_tot += (yi - y_mean) ** 2
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def print_results(x, y, a, b):
    preds = predict(a, b, x)
    resids = [yi - yp for yi, yp in zip(y, preds)]
    print(f"\nFitted line: y = {a:.6f} x + {b:.6f}")
    print(f"R^2 = {r_squared(x, y, a, b):.6f}\n")
    print(f"{'x':>6} {'y':>8} {'pred':>10} {'resid':>10}")
    for xi, yi, yp, r in zip(x, y, preds, resids):
        print(f"{xi:6.3f} {yi:8.3f} {yp:10.3f} {r:10.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit y = a*x + b to data provided interactively or via command-line")
    p.add_argument("--x", help="x values as comma- or space-separated numbers, e.g. '1,2,3'")
    p.add_argument("--y", help="y values as comma- or space-separated numbers, must match number of x values")
    p.add_argument("--plot", action="store_true", help="Show a plot of data and fitted line (requires matplotlib)")
    p.add_argument("--example", action="store_true", help="Use the example dataset x=[1,2,3,4,5], y=[14,13,9,5,2]")
    args = p.parse_args(argv)

  if args.example:
        x = [1, 2, 3, 4, 5]
        y = [14, 13, 9, 5, 2]
    else:
        if args.x and args.y:
            try:
                x = parse_sequence(args.x)
                y = parse_sequence(args.y)
            except ValueError as e:
                print(f"Error parsing input: {e}")
                sys.exit(1)
        else:
            # interactive prompts
            try:
                raw_x = input("Enter x values (comma- or space-separated): ").strip()
                raw_y = input("Enter y values (comma- or space-separated): ").strip()
                x = parse_sequence(raw_x)
                y = parse_sequence(raw_y)
            except (EOFError, KeyboardInterrupt):
                print("\nInput cancelled")
                sys.exit(1)
            except ValueError as e:
                print(f"Error parsing input: {e}")
                sys.exit(1)

  if len(x) == 0 or len(y) == 0:
        print("No data provided; exiting")
        sys.exit(1)
    if len(x) != len(y):
        print("x and y must have the same number of values")
        sys.exit(1)

  try:
        a, b = linear_fit(x, y)
    except ValueError as e:
        print(f"Could not fit data: {e}")
        sys.exit(1)

  print_results(x, y, a, b)

  if args.plot:
        try:
        
  import matplotlib.pyplot as plt
            xs = list(x)
            ys = list(y)
            ys_pred = predict(a, b, xs)
            plt.scatter(xs, ys, label='data')
            plt.plot(xs, ys_pred, color='red', label=f'fit: y={a:.3f}x+{b:.3f}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title('Linear fit')
            plt.show()
        except Exception:
            print("(matplotlib not available or failed to show plot)")


if __name__ == '__main__':
    main()
