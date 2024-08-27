import matplotlib.pyplot as plt
import numpy as np

# Data for ablation study
trapped_count = [1, 2, 3]
# sarsa_lambda_replacing = [1.91, 1.96, 3.21]
sarsa_lambda_cumulative = [4.52, 9.54, 7.62]
sarsa_5_sr = [2.31, 2.48, 5.62]
sr_llrl = [2.17, 4.37, 3.98]
dyna_sr = [2.63, 7.09, 4.62]

# Plotting
fig, ax = plt.subplots()

bar_width = 0.15
index = np.arange(len(trapped_count))

# Plotting each set of bars
# bar1 = ax.bar(index, sarsa_lambda_replacing, bar_width, label='SARSA(lambda)-Replacing')
bar2 = ax.bar(index + bar_width, sarsa_lambda_cumulative, bar_width, label='SR-SARSA(λ)')
bar3 = ax.bar(index + 2 * bar_width, sarsa_5_sr, bar_width, label='SR-SARSA(5)')
bar4 = ax.bar(index + 3 * bar_width, sr_llrl, bar_width, label='SR-LLRL')
bar5 = ax.bar(index + 4 * bar_width, dyna_sr, bar_width, label='SR-Dyna')

# Adding labels and title
ax.set_xlabel('Trapped Count')
ax.set_ylabel('Return')
ax.set_title('Ablation Study on Trapped Count')
ax.set_xticks(index + 2 * bar_width)
ax.set_xticklabels(trapped_count)

# Adding legend to the top left
ax.legend(loc='upper left')

# Style adjustments for academic style
plt.style.use('ggplot')
plt.grid(True, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.yaxis.set_tick_params(width=1.2)
ax.xaxis.set_tick_params(width=1.2)


# plot save:
plt.savefig('ablation_on_trapped_count.png', dpi=300, bbox_inches='tight')
# Show plot
# plt.show()
