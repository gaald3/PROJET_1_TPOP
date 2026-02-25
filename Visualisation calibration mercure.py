import pandas as pd
import matplotlib.pyplot as plt


file_path = '/Users/mac/Library/Mobile Documents/com~apple~CloudDocs/Session H26/TPOP Projet 1 /Exp 20 fev/calibration_mercure_1.TXT'
data = pd.read_csv(file_path, sep=',', usecols=[1, 2], names=['Pixel', 'Intensity'], header=None)

plt.figure(figsize=(12, 6))
plt.scatter(data['Pixel'], data['Intensity'], s=1, color='blue', alpha=0.5)
plt.xlabel('Position (Pixel)')
plt.ylabel('Intensité (comptes)')
plt.title('Visualisation du Spectre de Référence')
plt.savefig('ref_spectre_plot.png')
plt.show()