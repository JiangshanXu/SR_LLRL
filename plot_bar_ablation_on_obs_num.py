import matplotlib.pyplot as plt
import numpy as np

# Data
obs_nu = [100, 80, 60, 40, 20]
SR_Sarsa_lambda = [7.56, 6.28, 9.12, 10.87, 13.12]
SR_DynaQ = [3.24, 5.47, 8.12, 15.03, 16.11]
SR_Multi_step_Sarsa = [1.32, 2.98, 4.62, 8.36, 15.94]
SR_LLRL = [1.39, 3.31, 3.13, 8.91, 13.98]

# Convert None to np.nan for handling missing data
SR_Sarsa_lambda = [np.nan if v is None else v for v in SR_Sarsa_lambda]
SR_DynaQ = [np.nan if v is None else v for v in SR_DynaQ]
SR_Multi_step_Sarsa = [np.nan if v is None else v for v in SR_Multi_step_Sarsa]
SR_LLRL = [np.nan if v is None else v for v in SR_LLRL]

# Plotting
fig, ax = plt.subplots()

bar_width = 0.2
index = np.arange(len(obs_nu))

# Plotting each set of bars
bar1 = ax.bar(index, SR_Sarsa_lambda, bar_width, label='SR-Sarsa(λ)')
bar2 = ax.bar(index + bar_width, SR_DynaQ, bar_width, label='SR-DynaQ')
bar3 = ax.bar(index + 2 * bar_width, SR_Multi_step_Sarsa, bar_width, label='SR-Sarsa(5)')
bar4 = ax.bar(index + 3 * bar_width, SR_LLRL, bar_width, label='SR-LLRL')

# Adding labels and title
ax.set_xlabel('Obstacles')
ax.set_ylabel('Return')
ax.set_title('Ablation Study On Different Number of Obstacles')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(obs_nu)

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
plt.savefig('ablation_on_obs_num.png', dpi=300, bbox_inches='tight')

# Show plot
# plt.show()
