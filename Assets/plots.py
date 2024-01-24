import matplotlib.pyplot as plt

def plot_pid_mse(mse_history, pid_history):
    fig1 = plt.figure(1)
    fig1.suptitle("MSE History")
    plt.plot(mse_history)
    plt.xlabel("Time")
    plt.ylabel("Error (mse)")
    #Plot param history
    fig2 = plt.figure(2)
    fig2.suptitle("PID History")
    plt.plot(pid_history)
    plt.xlabel("Time")
    plt.ylabel("Param")
    plt.legend(["Kp", "Ki", "Kd"])
    #Show plot
    plt.show(block=False)
    plt.pause(1)
    input()
    plt.close()

def plot_mse(mse_history):
    fig1 = plt.figure(1)
    fig1.suptitle("MSE History")
    plt.plot(mse_history)
    plt.xlabel("Time")
    plt.ylabel("Error (mse)")
    #Show plot
    plt.show(block=False)
    plt.pause(1)
    input()
    plt.close()