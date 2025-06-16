#!/usr/bin/env python3
"""
plot_viscosity_vs_chain_length.py
---------------------------------
Modified version for analyzing viscosity files in specific directory structure.
Generates plots and saves them as PNG files instead of showing.

File structure: /public/home/ac9h310k3a/Gromacs/PMMA_250424/chain_length/CL/i/vis_new/viscosity.xvg
where i ranges from 1 to 8.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，必须在导入pyplot之前
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
import os

def load_inv_eta_xvg(xvg_path):
    """Return time (ps) and 1/viscosity arrays from an .xvg file."""
    t, inv_eta = [], []
    with open(xvg_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            try:
                time_ps, value = map(float, line.split()[:2])
                t.append(time_ps)
                inv_eta.append(value)
            except ValueError:
                continue
    if not t:
        raise ValueError(f"No numeric data found in {xvg_path}")
    return np.asarray(t), np.asarray(inv_eta)

def block_average(data):
    """
    Perform block averaging to find the optimal error estimate.
    Returns (mean, error).
    """
    data_mean = np.mean(data)
    if len(data) < 4:
        data_err = np.std(data, ddof=1) / np.sqrt(len(data))
        return data_mean, data_err

    n_points = len(data)
    max_block_size = n_points // 2
    if max_block_size < 1:
        return data_mean, np.std(data, ddof=1) / np.sqrt(n_points)
        
    block_sizes = np.logspace(0, np.log10(max_block_size), num=50, dtype=int)
    block_sizes = np.unique(block_sizes)

    block_errors = []
    for size in block_sizes:
        n_blocks = n_points // size
        if n_blocks < 2:
            break
        block_data = data[:n_blocks * size].reshape(n_blocks, size)
        block_means = np.mean(block_data, axis=1)
        sem_of_blocks = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        block_errors.append(sem_of_blocks)

    final_error = block_errors[-1] if block_errors else (np.std(data, ddof=1) / np.sqrt(len(data)))
    return data_mean, final_error

def cumulative_average(data):
    """Calculates the cumulative (running) average of a 1D array."""
    return np.cumsum(data) / np.arange(1, len(data) + 1)

def plot_scaling_law(chain_lengths, viscosities, viscosity_errors, output_file="scaling_law.png"):
    """Creates a log-log plot of viscosity vs molecular weight and fits the data."""
    
    # 准备数据 - 将链长转换为分子量 (M = N * 100)
    N = np.array(chain_lengths)
    M = N * 100  # 分子量 = 链长 × 100
    eta = np.array(viscosities)
    eta_err = np.array(viscosity_errors)
    
    log_M = np.log10(M)  # 使用分子量而不是链长
    log_eta = np.log10(eta)
    # 误差传递: δ(log10(x)) = δx / (x * ln(10))
    log_eta_err = eta_err / (eta * np.log(10))

    # 线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(log_M, log_eta)
    alpha = slope # 斜率就是标度指数 α
    
    # 创建新图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制带误差棒的散点图
    ax.errorbar(log_M, log_eta, yerr=log_eta_err, fmt='o', color='crimson',
                ecolor='lightcoral', elinewidth=2, capsize=4, markersize=8,
                label='Simulation Data')
    
    # 绘制拟合线
    fit_line = alpha * log_M + intercept
    ax.plot(log_M, fit_line, '-', color='k', lw=2,
            label=f'Fit: α = {alpha:.2f} ± {std_err:.2f}\n$R^2$ = {r_value**2:.3f}')
    
    # 配置图表
    ax.set_title('Viscosity Scaling Law (η ∝ M^α)', fontsize=16)
    ax.set_xlabel('log$_{10}$(Molecular Weight, M / g/mol)', fontsize=12)
    ax.set_ylabel('log$_{10}$(Viscosity, η / mPa·s)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scaling law plot saved as: {output_file}")
    
    # 打印分子量和拟合结果
    print(f"Molecular weight range: {M.min()} - {M.max()} g/mol")
    print(f"Scaling exponent α: {alpha:.3f} ± {std_err:.3f}")
    print(f"Correlation coefficient R²: {r_value**2:.3f}")

def main():
    # 定义基础路径和链长映射
    base_path = "/public/home/ac9h310k3a/Gromacs/PMMA_250424/chain_length/CL"
    
    # 链长映射：CL目录编号 → 实际链长
    CHAIN_LENGTHS_MAP = {
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,
        6: 60,
        7: 80,

    }

    # 构建文件路径列表
    xvg_files = []
    for i in range(1, 8):
        file_path = Path(f"{base_path}/{i}/vis_new/viscosity.xvg")
        if file_path.exists():
            xvg_files.append((file_path, CHAIN_LENGTHS_MAP[i]))
        else:
            print(f"Warning: File not found: {file_path}")

    if not xvg_files:
        print("No viscosity files found!")
        return

    # 时间窗口设置
    T_START_PS, T_END_PS = 55000, 60000

    # 设置matplotlib样式
    plt.style.use('default')  # 使用默认样式避免依赖问题
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(xvg_files)))

    print("-" * 65)
    print("Part 1: Analyzing individual simulation trajectories...")
    print(f"Time range: {T_START_PS} - {T_END_PS} ps")
    print("-" * 65)

    # 用于存储最终结果的列表
    final_results = []

    for i, (xvg_file, chain_length_N) in enumerate(xvg_files):
        try:
            # 加载数据
            time_ps, inv_eta_Pa_s = load_inv_eta_xvg(xvg_file)
            mask = (time_ps >= T_START_PS) & (time_ps <= T_END_PS)
            time_ps_selection = time_ps[mask]
            inv_eta_selection = inv_eta_Pa_s[mask]

            if len(inv_eta_selection) == 0: 
                print(f"No data in time range for {xvg_file.name}")
                continue

            # 计算平均值和误差
            avg_inv_eta, error_inv_eta = block_average(inv_eta_selection)
            if abs(avg_inv_eta) < 1e-9:
                final_eta_mPa_s, final_error_mPa_s = np.inf, np.inf
            else:
                final_eta_Pa_s = 1.0 / avg_inv_eta
                final_eta_mPa_s = final_eta_Pa_s * 1e3
                final_error_Pa_s = (final_eta_Pa_s**2) * error_inv_eta
                final_error_mPa_s = final_error_Pa_s * 1e3

            # 存储结果
            final_results.append({
                "N": chain_length_N,
                "eta": final_eta_mPa_s,
                "eta_err": final_error_mPa_s,
                "file": xvg_file.name
            })
            
            print(f"Results for: CL{xvg_file.parent.parent.name} (N={chain_length_N})")
            print(f"  Final η: {final_eta_mPa_s:.3f} ± {final_error_mPa_s:.3f} mPa·s")

            # 绘制第一个图：逆粘度 vs 时间
            ax1.plot(time_ps_selection, inv_eta_selection, marker='o', ms=2, lw=0.5, 
                    alpha=0.7, label=f"N={chain_length_N}", color=colors[i])
            
            # 绘制第二个图：累积平均粘度 vs 时间
            cum_avg_inv_eta = cumulative_average(inv_eta_selection)
            with np.errstate(divide='ignore', invalid='ignore'):
                cum_avg_eta_mPa_s = (1.0 / cum_avg_inv_eta) * 1e3
            ax2.plot(time_ps_selection, cum_avg_eta_mPa_s, lw=1.5, alpha=0.9, 
                    color=colors[i], label=f"N={chain_length_N}")
            ax2.axhline(final_eta_mPa_s, color=colors[i], linestyle='--', lw=1.5, alpha=0.8)

        except Exception as e:
            print(f"Error processing file {xvg_file}: {e}")

    # 配置并保存第一部分的双图
    if len(final_results) > 0:
        ax1.set_title("Signal Quality: Inverse Viscosity vs. Time", fontsize=14)
        ax1.set_xlabel("Time / ps", fontsize=12)
        ax1.set_ylabel("Inverse Viscosity / (kg⁻¹·m·s)", fontsize=12)
        ax1.axhline(0, color='k', linestyle=':', linewidth=1.0)
        ax1.legend(fontsize=10, title="Chain Length")
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title("Convergence: Cumulative Average Viscosity", fontsize=14)
        ax2.set_xlabel("Time / ps", fontsize=12)
        ax2.set_ylabel("Viscosity / mPa·s", fontsize=12)
        ax2.legend(fontsize=10, title="Chain Length")
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle("Individual Simulation Analysis", fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("individual_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Individual analysis plot saved as: individual_analysis.png")

    # 第二部分：绘制标度律图
    print("\n" + "-" * 65)
    print("Part 2: Plotting Viscosity Scaling Law (log-log plot)...")
    print("-" * 65)
    
    if len(final_results) > 1:
        # 按链长排序结果
        final_results.sort(key=lambda r: r['N'])
        
        # 提取数据列
        chain_lengths = [r['N'] for r in final_results]
        viscosities = [r['eta'] for r in final_results]
        viscosity_errors = [r['eta_err'] for r in final_results]
        
        # 打印最终结果表
        print("\nFinal Results Summary:")
        print(f"{'Chain Length (N)':<15} {'Molecular Weight (g/mol)':<25} {'Viscosity (mPa·s)':<20} {'Error (mPa·s)':<15}")
        print("-" * 75)
        for r in final_results:
            molecular_weight = r['N'] * 100
            print(f"{r['N']:<15} {molecular_weight:<25} {r['eta']:<20.3f} {r['eta_err']:<15.3f}")
        
        # 调用函数绘制标度律图
        plot_scaling_law(chain_lengths, viscosities, viscosity_errors, "viscosity_scaling_law.png")
    else:
        print("Not enough data points to plot the scaling law.")

    print("\nAnalysis completed! Check the generated PNG files.")

if __name__ == "__main__":
    main()