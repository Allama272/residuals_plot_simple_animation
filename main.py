import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter

import mplcyberpunk

plt.style.use("cyberpunk")
np.random.seed(seed=52)

def vis(e, m):
    y_ax = e["y_axis"] - (e["x_axis"] * (1 - m))
    sns.scatterplot(x=e["x_axis"], y=y_ax, s=100, zorder=5)
    plt.axline((0, 0), (20, m * 20))
    plt.xlim(0, 12)
    plt.ylim(-5, 12)
    for i in range(len(e)):
        x, y = e.loc[i, "x_axis"], y_ax[i]
        plt.plot([x, x], [y, m * x],color='r', linestyle="--", zorder=1)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
def vis_line(e, m, l=-1, error_line=-1):
    y_ax = e["y_axis"] - (e["x_axis"] * (1 - m))
    sns.scatterplot(x=e["x_axis"], y=y_ax, s=100, zorder=5)
    if l >= 0:
        plt.plot((0, l), (0, l), linestyle='-')
    else:
        plt.axline((0, 0), (20, m * 20))
    plt.xlim(0, 12)
    plt.ylim(-5, 12)
    if error_line >= 0:
        for i in range(len(e)):
            x, y = e.loc[i, "x_axis"], y_ax[i]
            plt.plot([x, x], [y, (m * x) + (y - (m * x)) * error_line], color='r', linestyle="--", zorder=1)

    else:
        for i in range(len(e)):
            x, y = e.loc[i, "x_axis"], y_ax[i]
            plt.plot([x, x], [y, m * x], color='r', linestyle="--", zorder=1)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

metadata = dict(title="variance_graph", authoer="allama")
#writer = FFMpegWriter(fps=30, metadata=metadata)
writer = PillowWriter(fps=30, metadata=metadata)

fig, ax=plt.subplots(figsize=(20, 9))
fig.subplots_adjust(top=0.8, bottom= 0.15)
ax.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
plt.style.use("cyberpunk")



x_values = np.linspace(1, 10, 100)

# Create y values initially following y = x
y_values_homo = x_values + 2*np.random.uniform(-3, 3, len(x_values))
y_values_hetro = x_values + 2*np.random.normal(0, 0.1, len(x_values))


for i in range(1, len(x_values)):
    y_values_hetro[i] += np.random.choice([-1, 1]) * np.random.uniform(1, 7) * i / len(x_values)

err_homo = pd.DataFrame({"x_axis": x_values,
                    "y_axis": y_values_homo})
err_hetro = pd.DataFrame({"x_axis": x_values,
                    "y_axis": y_values_hetro})
ho_text = "Homoscedasticity: Variance of the Errors is Constant \n Graphs Looks Like a Random Cloud of Noise"
he_text = "Hetroscedasticity: Variance of the Errors is *NOT* Constant \n Error Graph Has any Pattern"

time = 120
with writer.saving(fig, "Variance.gif",100):
    ### Intro
    plt.plot()
    plt.grid(False)
    line_time=np.arange(start=0, stop=12.5, step=0.1)
    for i in line_time:

        plt.subplot(1, 2, 1)

        vis_line(err_homo, round(1, 3), l = i, error_line=1)
        plt.xlim(0, 12)
        plt.ylim(-10, 20)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.xlabel("X Values", fontsize=15)
        plt.ylabel("True Y Values", fontsize=15)
        plt.legend(["predicted", "Regression Line", "Error"]);
        plt.title("Prediction Plot", fontsize=19);
        fig.text(.3, 0.03, ho_text, ha='center', fontsize=15, fontweight="bold")
        mplcyberpunk.make_scatter_glow()


        plt.subplot(1, 2, 2)
        vis_line(err_hetro, round(1, 3), l = i, error_line=1)
        plt.xlim(0, 12)
        plt.ylim(-10, 20)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.xlabel("X Values", fontsize=15)
        plt.ylabel("True Y Values", fontsize=15)
        plt.legend(["predicted", "Regression Line", "Error"]);
        plt.title("Prediction Plot", fontsize=19);
        fig.text(0.76, 0.03, he_text, ha='center', fontsize=15, fontweight="bold")
        plt.suptitle('Homoscedasticity vs Heteroscedasticity \n', fontsize=30)
        fig.text(0.5, 0.5, "Allama", ha='center', fontsize=20, fontweight="bold", rotation=90)

        mplcyberpunk.make_scatter_glow()
        writer.grab_frame()
        plt.clf()


    error_time=np.arange(start=0, stop=6.5, step=0.15)
    max_error = max(error_time)
    for i in error_time:

        plt.subplot(1, 2, 1)

        vis_line(err_homo, round(1, 3), error_line=1-(i/max_error))
        plt.xlim(0, 12)
        plt.ylim(-10, 20)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.xlabel("X Values", fontsize=15)
        plt.ylabel("True Y Values", fontsize=15)
        plt.legend(["predicted", "Regression Line", "Error"]);
        plt.title("Prediction Plot", fontsize=19);
        fig.text(.3, 0.03, ho_text, ha='center', fontsize=15, fontweight="bold")
        mplcyberpunk.make_scatter_glow()


        plt.subplot(1, 2, 2)
        vis_line(err_hetro, round(1, 3), error_line=1-(i/max_error))
        plt.xlim(0, 12)
        plt.ylim(-10, 20)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.xlabel("X Values", fontsize=15)
        plt.ylabel("True Y Values", fontsize=15)
        plt.legend(["predicted", "Regression Line", "Error"]);
        plt.title("Prediction Plot", fontsize=19);
        fig.text(0.76, 0.03, he_text, ha='center', fontsize=15, fontweight="bold")
        plt.suptitle('Homoscedasticity vs Heteroscedasticity \n', fontsize=30)
        fig.text(0.5, 0.5, "Allama", ha='center', fontsize=20, fontweight="bold", rotation=90)

        mplcyberpunk.make_scatter_glow()
        writer.grab_frame()
        plt.clf()
  #  for i in range(time):
  #      writer.grab_frame()
   # plt.clf()
    ##### Animation
    for i in np.arange(start=1, stop=-0.0005, step=-0.01):
    #for i in range(1):
        plt.subplot(1,2,1)
        vis(err_homo,  round(i, 3))
        plt.xlim(0, 12)
        plt.ylim(-10, 20)
        plt.xlabel("")
        plt.ylabel("")

        fig.text(.3, 0.03, ho_text, ha='center', fontsize=15, fontweight="bold")
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        mplcyberpunk.make_scatter_glow()



        plt.subplot(1, 2, 2)
        vis(err_hetro, round(i, 3))
        plt.xlim(0, 12)
        plt.ylim(-10, 20)

        plt.xlabel("")
        plt.ylabel("")
        fig.text(0.76, 0.03, he_text, ha='center', fontsize=15, fontweight="bold")
        mplcyberpunk.make_scatter_glow()
        plt.suptitle('Homoscedasticity vs Heteroscedasticity \n', fontsize=30)
        fig.text(0.5, 0.5, "Allama", ha='center', fontsize=20, fontweight="bold", rotation=90)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        writer.grab_frame()

        plt.clf()
    ### End
    plt.subplot(1, 2, 1)
    vis(err_homo, 0)
    plt.xlim(0, 12)
    plt.ylim(-10, 20)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.xlabel("X Values", fontsize=15)
    plt.ylabel("Error Values", fontsize=15)
    plt.title("Error Plot", fontsize=19);
    fig.text(.3, 0.03, ho_text, ha='center', fontsize=15, fontweight="bold")
    mplcyberpunk.make_scatter_glow()

    plt.subplot(1, 2, 2)
    vis(err_hetro, 0)
    plt.xlim(0, 12)
    plt.ylim(-10, 20)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.xlabel("X Values", fontsize=15)
    plt.ylabel("Error Values", fontsize=15)
    plt.title("Error Plot", fontsize=19)
    fig.text(0.76, 0.03, he_text, ha='center', fontsize=15, fontweight="bold")
    plt.suptitle('Homoscedasticity vs Heteroscedasticity \n', fontsize=30)
    fig.text(0.5, 0.5, "Allama", ha='center', fontsize=20, fontweight="bold", rotation=90)

    mplcyberpunk.make_scatter_glow()
    for i in range(time):
        writer.grab_frame()
