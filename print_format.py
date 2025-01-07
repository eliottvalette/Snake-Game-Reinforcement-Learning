import numpy as np

def print_box(episode, metrics, last_score, last_steps, epsilon):
    avg_score = np.mean(last_score)
    avg_steps = np.mean(last_steps)
    max_score = np.max(last_score)
    max_steps = np.max(last_steps)
    box_width = 60
    separator = "+" + "-" * (box_width - 2) + "+"
    print(separator)
    print(f"| Episode: {episode:<50} |")
    print(separator)
    print(f"| {'Avg Score':<20}: {avg_score:<30.4f} |")
    print(f"| {'Max Score':<20}: {max_score:<30.4f} |")
    print(f"| {'Avg Steps':<20}: {avg_steps:<30.4f} |")
    print(f"| {'Max Steps':<20}: {max_steps:<30.4f} |")
    print(f"| {'Epsilon':<20}: {f'{epsilon * 100:.4f}%':<30} |")  # Include `%` in formatted string
    print(separator)

    for key, value in metrics.items():
        print(f"| {key:<20}: {value:<30.4f} |")
    print(separator)

