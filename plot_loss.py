#! /usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd

def parse_experiment_log(filename):
    cumulative_time = 0
    times = []
    losses = []
    
    with open(filename, 'r') as f:
        start_parsing = False
        for line in f:
            line = line.strip()
            # Identify the header line to start parsing the data below it
            if line.startswith("Step"):
                start_parsing = True
                continue
            
            if start_parsing:
                parts = line.split()
                # Ensure the line has enough columns and starts with a step number
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        step = int(parts[0])
                        loss = float(parts[1])
                        step_time_ms = float(parts[2])

                        if step == 1:
                           step_time_ms = 0

                        # Accumulate time (convert ms to seconds for better readability)
                        cumulative_time += (step_time_ms / 1000.0)
                        
                        times.append(cumulative_time)
                        losses.append(loss)
                    except ValueError:
                        continue
                        
    return times, losses

tuned = os.environ.get('tuned', 'False') == 'True'

if tuned:
    files = {
        'Malbo (tuned)': 'malbo.lr0p75x.out',
        'Baseline (tuned)': 'baseline.lr0p5x.out',
    }
else:
    files = {
        'Malbo': 'malbo.out',
        'Baseline': 'baseline.out'
    }

plt.figure(figsize=(10, 6))

min_y_loss = 10
for label, filename in files.items():
    x_time, y_loss = parse_experiment_log(filename)
    min_y_loss = min(min_y_loss, min(y_loss))

def format_loss(y, pos):
    return f'{y + min_y_loss:g}'

for label, filename in files.items():
    try:
        x_time, y_loss = parse_experiment_log(filename)
        df = pd.DataFrame({'time': x_time, 'loss': y_loss})
        line, = plt.plot(df['time'], df['loss'] - min_y_loss, alpha=0.2)
        smoothed_loss = df['loss'].rolling(window=100, min_periods=1).mean()
        plt.plot(df['time'], smoothed_loss - min_y_loss, linewidth=2, label=label, color=line.get_color())

        if label.startswith('Baseline'):
            plt.axhline(smoothed_loss.iloc[-1] - min_y_loss, linestyle='-', linewidth=2.0, alpha=0.6, color=line.get_color())
    except FileNotFoundError:
        print(f"File {filename} not found.")

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_loss))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(format_loss))

plt.xlabel('Total Elapsed Time (seconds)')
plt.ylabel('Loss')
plt.ylim([3.02 - min_y_loss, 4 - min_y_loss])
plt.title('Training Loss vs. Total Elapsed Time')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('loss_plot.tunedlr.png' if tuned else 'loss_plot.png')
plt.show()
