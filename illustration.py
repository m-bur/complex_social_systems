import pandas as pd
from utils.measure import *

# df_stats = pd.read_csv('illustration/media_statistics.csv')
# df_results = pd.read_csv('illustration/network_polarization.txt')
# df_results.iloc[:,0] = df_results.iloc[:,0].apply(lambda x: 1 if x > 0 else -1)
# df_election = df_results.iloc[::4*365,0]
#df = get_consecutive_terms_counts(df_election.to_list())


# df1 = pd.read_csv('illustration/consecutive_terms_1.txt', sep=';', header=None)
# df1.iloc[:,1] = df1.iloc[:,1] + 0.4
# df1.iloc[:, :] = df1.iloc[:, :].round()
# df1.iloc[:,0] = df1.iloc[:,0] // (7*4)
# df1 = df1.set_index(df1.columns[0])
# df1 = df1.groupby(df1.index).sum()
# df2 = pd.read_csv('illustration/consecutive_terms_2.txt',sep=';', header=None)
# df2.iloc[:, :] = df2.iloc[:, :].round()
# df2.iloc[:,0] = df2.iloc[:,0] // (7*4)
# df2 = df2.set_index(df2.columns[0])
# df2 = df2.groupby(df2.index).sum()
# df=pd.concat([df1, df2], axis=1).fillna(0)


# df = pd.read_csv('illustration/opinion_trend.txt',sep='\t')
# df = df[['-1','1']]
# df = df.div(df.sum(axis=1), axis=0)

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX style
plt.rc('text.latex', preamble=r'\usepackage{sfmath}')  # Use sans-serif math mode
plt.rcParams.update({
    'font.size': 18,  # General font size
    'axes.titlesize': 20,  # Title font size
    'axes.labelsize': 20,  # Label font size
    'legend.fontsize': 16,  # Legend font size
    'xtick.labelsize': 16,  # x-axis tick font size
    'ytick.labelsize': 16   # y-axis tick font size
})
# plt.figure(figsize=(8, 6))
# # Iterate over the columns (opinions) in the DataFrame
# for column in df.columns:
#     # Assign colors based on the opinion value
#     if column == '-1':
#         color = 'blue'
#     elif column == '1':
#         color = 'red'
#     else:
#         color = 'grey'

#     # Plot the opinion share over time
#     plt.plot(df.index * 5/365, df[column],
#                  label=f"Opinion {column}", color=color, linewidth=1.75)

# plt.axvline(x=10, color='k', linestyle='--', linewidth=1.9)
# plt.xlim([0, 100])
# plt.ylim([0, 0.65])
# # Label the x-axis (time)
# plt.xlabel(r"$t \; (years)$")

# # Label the y-axis (opinion share)
# plt.ylabel(r"Voter Share")

# plt.grid()
# plt.minorticks_on()

# # Customize the grid
# plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.4)  # Major grid
# plt.grid(which='minor', color='black', linestyle='--', linewidth=0.3, alpha=0.4)  # Minor grid
# # Display the legend
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.savefig('illustration/voter_share.png')

# # Plot the histogram
# plt.figure(figsize=(8, 6))
# plt.bar(df.index+0.2, df.iloc[:,0], width=0.4, color="red", label="Opinion 1", align="center", alpha=0.8)
# plt.bar(df.index-0.2, df.iloc[:,1], width=0.4, color="blue", label="Opinion -1", align="center", alpha=0.8)

# # Add titles and labels
# plt.xticks(ticks=range(int(df.index.min()), int(df.index.max())+1))
# plt.yticks(ticks=range(0, 7, 1))
# plt.ylim([0, 6])
# #plt.title("Histogram of consecutive terms")
# plt.xlabel("Number of consecutive terms")
# plt.ylabel("Frequency")
# plt.legend(loc="upper right")
# plt.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.6)
# plt.savefig('illustration/consecutive_terms_1.pdf')
# plt.savefig('illustration/consecutive_terms_1.png')
# x_values = df_stats.index / 365
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(x_values, df_stats["mean"], label="Mean", color="black", linestyle="--")
# ax.fill_between(
#         x_values,
#         df_stats["mean"] - df_stats["std"],
#         df_stats["mean"] + df_stats["std"],
#         color="grey",
#         alpha=0.75,
#         label="Mean ± Std"
# )
# for i in range(len(df_election)-1):
#     color = "blue" if df_election.iloc[i] == -1 else "red"
#     ax.axvspan(df_election.index[i] / 365, df_election.index[i+1] / 365, color=color, alpha=0.2)
# color = "blue" if df_election.iloc[-1] == -1 else "red"
# ax.axvspan(df_election.index[-1] / 365, x_values[-1], color=color, alpha=0.0875)
# plt.xlim([0, 100])
# ax.set_xlabel(r"$t \; (years)$")
# ax.set_ylabel("Media Opinion")
# ax.legend()
# # ax.set_ylim(min(df_stats["mean"]+df_stats["std"]), max((df_stats["mean"] + df_stats["std"]).max()) * 1.1)
# plt.tight_layout()
# plt.savefig('illustration/media_opinion.png')
# plt.savefig('illustration/media_opinion.pdf')