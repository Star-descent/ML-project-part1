import json
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
runs = root / 'runs'
names = ['adamw-default', 'adamw-other', 'muon-default', 'muon-other', 'sgdm-default', 'sgdm-other']
rows = []
for name in names:
    config = json.loads((runs / name / 'config.json').read_text())
    summary = json.loads((runs / name / 'summary.json').read_text())
    optimizer = config['optimizer_name']
    rate = config[{'adamw': 'learning_rate', 'muon': 'muon_learning_rate', 'sgdm': 'sgdm_learning_rate'}[optimizer]]
    rows.append((name, optimizer, rate, summary['best_val_loss'], summary['endpoint_scores']['loss'], summary['endpoint_scores']['perplexity']))

report_path = runs / 'part1-report.pdf'
with PdfPages(report_path) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, 'Part I: GPT Pre-training Optimizer Comparison', fontsize=18, weight='bold')
    fig.text(0.08, 0.90, 'Dataset: Shakespeare character-level modeling; model: GPT-2 medium', fontsize=10)
    body = [
        'Objective',
        'Compare AdamW, Muon, and SGDM under the default peak learning rate and one alternative rate.',
        '',
        'Experimental protocol',
        'All runs used the same model, data, token budget (92,160,000 tokens), cosine schedule,',
        'batch size, dropout, seeds, and evaluation procedure. The alternative rates were AdamW 1.5e-4,',
        'Muon 5e-3, and SGDM 0.2.',
        '',
        'Results',
    ]
    y = 0.85
    for line in body:
        fig.text(0.08, y, line, fontsize=11, weight='bold' if line in ('Objective', 'Experimental protocol', 'Results') else 'normal')
        y -= 0.028
    y -= 0.01
    headers = ('Run', 'Optimizer', 'Peak LR', 'Best val loss', 'Final val loss', 'Final PPL')
    x_positions = [0.08, 0.31, 0.49, 0.64, 0.78, 0.91]
    for x, header in zip(x_positions, headers):
        fig.text(x, y, header, fontsize=8, weight='bold', ha='right' if x > 0.5 else 'left')
    y -= 0.027
    for name, optimizer, rate, best, endpoint, ppl in rows:
        values = (name, optimizer, f'{rate:g}', f'{best:.4f}', f'{endpoint:.4f}', f'{ppl:.2f}')
        for x, value in zip(x_positions, values):
            fig.text(x, y, value, fontsize=7.5, ha='right' if x > 0.5 else 'left')
        y -= 0.024
    conclusions = [
        '',
        'Interpretation',
        'SGDM at 0.1 achieved the lowest best validation loss (1.5168) and remained much more stable late in training.',
        'AdamW and Muon reached low validation loss early but then showed strong overfitting; their final losses',
        'were substantially higher than their best losses. Reducing AdamW to 1.5e-4 slightly improved its best',
        'validation loss from 1.5579 to 1.5521. Reducing Muon to 5e-3 did not improve its best loss.',
        'Increasing SGDM to 0.2 did not improve the best loss, but its final perplexity remained close to the default.',
        '',
        'Conclusion',
        'For this fixed training budget and configuration, SGDM with peak learning rate 0.1 is the strongest overall',
        'choice by validation stability and best validation loss. The best checkpoint should be used for evaluation',
        'when the validation curve rises after an earlier minimum.',
    ]
    y -= 0.02
    for line in conclusions:
        fig.text(0.08, y, line, fontsize=10, weight='bold' if line in ('Interpretation', 'Conclusion') else 'normal')
        y -= 0.027
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
print(report_path)
