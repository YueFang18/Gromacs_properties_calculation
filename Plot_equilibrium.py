import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# ====================
# 学术风格参数配置
# ====================
plt.rcParams.update({
    'font.family': 'sans-serif',  # 推荐使用Arial或Helvetica
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
    'axes.linewidth': 1.2,
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'savefig.dpi': 600,
    'savefig.format': 'tiff',
    'savefig.bbox': 'tight',
    'figure.constrained_layout.use': True
})

# ====================
# 数据文件配置
# ====================
file_config = [
    {'path': 'eq_potential.xvg', 'title': '(a) Potential Energy', 'ylabel': 'kJ/mol', 
     'color': '#2E86C1', 'cols': (0,1), 'ylim': None},
    
    {'path': 'eq_density.xvg',    'title': '(b) Density',          'ylabel': r'g/cm$^3$',  
     'color': '#E74C3C', 'cols': (0,1), 'ylim': None},
    
    {'path': 'eq_temperature.xvg','title': '(c) Temperature',      'ylabel': 'K',      
     'color': '#F1C40F', 'cols': (0,1), 'ylim': None},
    
    {'path': 'eq_pressure.xvg',   'title': '(d) Pressure',         'ylabel': 'bar',    
     'color': '#27AE60', 'cols': (0,1), 'ylim': (-50, 50)},
    
    {'path': 'eq_msd.xvg',        'title': '(e) MSD',             'ylabel': r'nm$^2$',    
     'color': '#8E44AD', 'cols': (0,1), 'ylim': None},
    
    {'path': 'eq_gyrate.xvg',     'title': '(f) Radius of Gyration','ylabel': 'nm',    
     'color': '#34495E', 'cols': (0,1), 'ylim': (8.00, 9.0)}  # 手动设置Rg范围
]

# ====================
# 创建画布与布局
# ====================
fig = plt.figure(figsize=(12, 8), dpi=600)
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.45)

# ====================
# 数据加载与绘图
# ====================
for idx, config in enumerate(file_config):
    ax = fig.add_subplot(gs[idx//3, idx%3])
    
    # 加载数据
    data = np.loadtxt(config['path'], comments=['#','@'], usecols=config['cols'])
    time = data[:, 0]/1000  # ps -> ns
    values = data[:, 1]
    
    # 绘制主数据线
    ax.plot(time, values, color=config['color'], 
            alpha=0.9, label='Raw Data')
    
    # 平衡线绘制（最后1/3数据）
    final_third = len(time)//3
    t_start = time[-final_third]
    mean_val = np.mean(values[-final_third:])
    std_val = np.std(values[-final_third:])
    
    ax.axhline(mean_val, color='#2C3E50', ls=':', lw=2, 
               label=r'$\mu = {:.2f} \pm {:.2f}$'.format(mean_val, std_val))
    ax.axvspan(90, 100, color='gray', alpha=0.1)  # 标记统计区域
    
    # MSD专用处理
    if 'MSD' in config['title']:
        # 线性拟合
        fit_start = int(0.2*len(time))
        x_fit = time[fit_start:]
        y_fit = values[fit_start:]
        coeffs = np.polyfit(x_fit, y_fit, 1)
        ax.plot(x_fit, coeffs[0]*x_fit + coeffs[1], '--', 
                color='#C0392B', lw=2, label='Linear fit')
        
    # 坐标轴设置
    ax.set_xlabel('Time (ns)', fontweight='semibold')
    ax.set_ylabel(config['ylabel'], fontweight='semibold')
    ax.set_title(config['title'], fontweight='bold', pad=12)
    
    # 手动设置y轴范围
    if config['ylim'] is not None:
        ax.set_ylim(config['ylim'][0], config['ylim'][1])
    else:
        y_range = np.ptp(values[-final_third:])
        ax.set_ylim(mean_val-1.5*y_range, mean_val+1.5*y_range)
    
    # 图例与网格
    ax.legend(loc='lower right' if 'MSD' in config['title'] else 'best', 
              frameon=True, framealpha=0.9, facecolor='white')
    ax.grid(True, alpha=0.4)

# ====================
# 保存输出
# ====================
plt.savefig('Equilibrium_Analysis.tiff', pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("学术风格图表已保存为 Equilibrium_Analysis.tiff")
