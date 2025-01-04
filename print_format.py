import textwrap

def print_box(episode, metrics):
    box_width = 60
    separator = "+" + "-" * (box_width - 2) + "+"
    print(separator)
    print(f"| Episode: {episode:<51} |")
    print(separator)

    for key, value in metrics.items():
        print(f"| {key:<20}: {value:<30.4f} |")
    print(separator)
