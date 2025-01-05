def print_box(episode, metrics, avg_score, avg_steps, epsilon):
    box_width = 60
    separator = "+" + "-" * (box_width - 2) + "+"
    print(separator)
    print(f"| Episode: {episode:<50} |")
    print(separator)
    print(f"| {'Avg Score':<20}: {avg_score:<30.4f} |")
    print(f"| {'Avg Steps':<20}: {avg_steps:<30.4f} |")
    print(f"| {'Epsilon':<20}: {f'{epsilon * 100:.4f}%':<30} |")  # Include `%` in formatted string
    print(separator)

    for key, value in metrics.items():
        print(f"| {key:<20}: {value:<30.4f} |")
    print(separator)

