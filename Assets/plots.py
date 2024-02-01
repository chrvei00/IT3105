import matplotlib.pyplot as plt

def plot_pid_mse(mse_history, pid_history, filename):
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # MSE History plot
    ax1.set_title("MSE History")
    ax1.plot(mse_history)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Error (mse)")

    # PID History plot
    ax2.set_title("PID History")
    ax2.plot(pid_history)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Param")
    ax2.legend(["Kp", "Ki", "Kd"])

    # Adjust layout and save the plot
    plt.tight_layout()

    # Show plot
    plt.draw()
    plt.pause(0.001)

    # Save plot
    if input("\nSave plot? (y/n) ") == "y":
        plt.savefig(f"plots/{filename}.png")
    # Close plot
    plt.close()

def plot_mse(mse_history, filename):
    fig1 = plt.figure(1)
    fig1.suptitle("MSE History")
    plt.plot(mse_history)
    plt.xlabel("Time")
    plt.ylabel("Error (mse)")
    #Show plot
    plt.draw()
    plt.pause(0.001)
    # Save plot
    if input("\nSave plot? (y/n) ") == "y":
        plt.savefig(f"plots/{filename}.png")
    # Close plot
    plt.close()