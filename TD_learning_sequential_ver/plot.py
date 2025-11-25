import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("2048_scores.txt", "r") as file:
        scores = [int(line.strip()) for line in file.readlines()]
    plt.plot(scores, marker='o', linestyle='-', color='b')
    plt.title("2048 Game Scores Over Time")
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()