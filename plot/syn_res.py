# compare performance on syn data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
method = ['mlp', 'gat', 'gcn']
color = ['#6A5DC4', '#D2AA3A', '#47AF79', '#3B5387', '#D94738']
shade = ['#E9E6F7', '#FAF1DE', '#E2F4E9', '#C5CCDB', '#FCD1CC']
label = ['MLP', 'GAT', 'GCN']
df = pd.read_csv('../syn_baseline_new/syn_results/syn_results_dropout0.1.csv', header=None, usecols=[0, 1, 2, 3])
df.columns = ['method', 'h', 'mean', 'std']
wd=0.8; ft=20; lft=8

fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

for i in range(len(method)):
    res = df[df['method'] == method[i]]
    sorted_res = res.sort_values(by='h')
    res_mean = sorted_res['mean'].to_numpy()
    res_std = sorted_res['std'].to_numpy()
    # Plot the line
    ax1.plot(h, res_mean, label=label[i], color = color[i], linewidth=wd)
    # Add shaded area for standard deviation
    # plt.fill_between(
    #     h, res_mean - res_std, res_mean + res_std, 
    #     color=shade[i], alpha=0.6, linewidth=wd)

# dgcn
dgcn = pd.read_csv('../syn_baseline_new/syn_results/syn_results_GCN_l2.csv', header=None, usecols=[0, 1, 2, 3, 4])
dgcn.columns = ['method', 'h', 'gamma', 'mean', 'std']
dgcn_mean = []; dgcn_std = []; dgcn_optim_gamma = []
dgcn_mean_low = []; dgcn_std_low = []; dgcn_wrost_gamma = []
for hi in h:
    res = dgcn[dgcn['h'] == hi]
    max_row = res[res['mean'] == res['mean'].max()]
    dgcn_mean.append(max_row['mean'].values[0])
    dgcn_std.append(max_row['std'].values[0])
    dgcn_optim_gamma.append(max_row['gamma'].values[0])
    min_row = res[res['mean'] == res['mean'].min()]
    dgcn_mean_low.append(min_row['mean'].values[0])
    dgcn_std_low.append(min_row['std'].values[0])
    dgcn_wrost_gamma.append(min_row['gamma'].values[0])

dgcn_mean = np.array(dgcn_mean); dgcn_std = np.array(dgcn_std)
dgcn_mean_low = np.array(dgcn_mean_low); dgcn_std_low = np.array(dgcn_std_low)

ax1.plot(h, dgcn_mean, label='PD-GCN (optimal $\gamma$)', color = color[3], linewidth=wd)
# Add shaded area for standard deviation
# plt.fill_between(
#     h, dgcn_mean - dgcn_std, dgcn_mean + dgcn_std, 
#     color=shade[3], alpha=0.6, linewidth=wd)
ax1.plot(h, dgcn_mean_low, label='PD-GCN (worst $\gamma$)', color = color[3], linewidth=wd, linestyle='dotted')

# dgat
dgat = pd.read_csv('../syn_new_exp_res/syn_critical_norewire_with_selfloop_0.01.csv', header=None, usecols=[0, 1, 2, 3, 4, 5])
dgat.columns = ['method', 'h', 'gamma', 'alpha', 'mean', 'std']
dgat_mean = []; dgat_std = []; dgat_optim_gamma = []
dgat_mean_low = []; dgat_std_low = []; dgat_wrost_gamma = []
for hi in h:
    res = dgat[dgat['h'] == hi]
    max_row = res[res['mean'] == res['mean'].max()]
    dgat_mean.append(max_row['mean'].values[0])
    dgat_std.append(max_row['std'].values[0])
    dgat_optim_gamma.append(max_row['gamma'].values[0])
    min_row = res[res['mean'] == res['mean'].min()]
    dgat_mean_low.append(min_row['mean'].values[0])
    dgat_std_low.append(min_row['std'].values[0])
    dgat_wrost_gamma.append(min_row['gamma'].values[0])

dgat_mean = np.array(dgat_mean); dgat_std = np.array(dgat_std)
dgat_mean_low = np.array(dgat_mean_low); dgat_std_low = np.array(dgat_std_low)

ax1.plot(h, dgat_mean, label='PD-GAT (optimal $\gamma$)', color = color[4], linewidth=wd)
# Add shaded area for standard deviation
# plt.fill_between(
#     h, dgat_mean - dgat_std, dgat_mean + dgat_std, 
#     color=shade[4], alpha=0.6, linewidth=wd)
ax1.plot(h, dgat_mean_low, label='PD-GAT (worst $\gamma$)', color = color[4], linewidth=wd, linestyle='dotted')

ax1.set_xlabel("Homophily", fontsize=ft)
ax1.set_ylabel("Accuracy", fontsize=ft)
x_ticks = np.arange(0.0, 1.0, 0.1)
ax1.set_xticks(x_ticks)
ax1.legend(loc="upper left", fontsize=lft)

plt.tight_layout()
plt.savefig('syn_res1.eps', format='eps', dpi=1000)  # dpi=300 for high resolution

# compare optimal gamma and h for 1-layer DGCN

fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))

h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dgcn1 = pd.read_csv('../syn_baseline_new/syn_results/syn_results_GCN_l1.csv', header=None, usecols=[0, 1, 2, 3, 4])
dgcn1.columns = ['method', 'h', 'gamma', 'mean', 'std']

# Select a colormap (e.g., 'viridis')
cmap = plt.get_cmap('viridis')

# Generate 10 evenly spaced colors from the colormap
colors = [cmap(i) for i in np.linspace(0, 1, 10)]

for i in range(len(h)):
    hi = h[i]
    view = dgcn1[dgcn1['h'] == hi]
    max_row = view[view['mean'] == view['mean'].max()]
    ax2.plot(view['gamma'].values, view['mean'].values, 'k', color=colors[i], linewidth=wd)
    ax2.scatter(max_row['gamma'].values, max_row['mean'].values, 
        color=colors[i], marker='o')


# Normalize the colormap to the number of lines (10 in this case)
norm = mpl.colors.Normalize(vmin=0, vmax=9)

# Create a ScalarMappable object to use with the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

# Add the colorbar to the plot
cbar = plt.colorbar(sm)
cbar.set_label('Homophily')
cbar.set_ticks(np.arange(0, 10, 1))  # Ticks at 0, 1, 2, ..., 9
cbar.set_ticklabels([f'{i/10:.1f}' for i in range(10)])

ax2.set_xlabel("$\gamma$", fontsize=ft)
ax2.set_ylabel("Accuracy", fontsize=ft)
x_ticks = np.arange(0.1, 1.1, 0.1)
ax2.set_xticks(x_ticks)

plt.tight_layout()
plt.savefig('syn_res2.png')
plt.savefig('syn_res2.eps', format='eps', dpi=1000)  # dpi=300 for high resolution

