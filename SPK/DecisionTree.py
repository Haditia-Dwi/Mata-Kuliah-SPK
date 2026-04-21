import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv('Dataset_Kelayakan_Kredit.csv')
x = df.drop(columns=['Status'])
y = df['Status']

model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

model.fit(x, y)
print("Model berhasi di Training. Kini model sudah siap untuk digunakan")

print("Sedang menggambar pohon keputusan")

plt.figure(figsize=(14,8))
tree.plot_tree(model,
               feature_names=['Pendapatan_Juta', 'Riwayat_Baik', 'Ada_Jaminan'],
               class_names=model.classes_,
               filled=True,
               rounded=True,
               fontsize=10)

plt.title("Visualisasi Pohon Keputusan Kelayakan Kredit")
plt.savefig("Pohon_keputusan.png")
print("Berhasil, gambar pohon keputusan telah disimpan sebagai 'Pohon_Keputusan.png'.\n")

print("== SISTEM KEPUTUSAN UNTUK MENENTUKAN KELAYAKAN KREDIT")
input_pendapatan = float(input('Masukkan pendapatan calon nasabah: '))
teks_riwayat = input("Masukkan riwayat kredit calon nasabah(ada/tidak): ")
teks_jaminan = input("Masukkan jaminan yang dimiliki oleh calon nasabah(ada/tidak): ")

if teks_riwayat == "ya" or teks_riwayat == "ada":
    input_riwayat = 1
else:
    input_riwayat = 0

if teks_jaminan == "ya" or teks_jaminan == "ada":
    input_jaminan = 1
else:
    input_jaminan = 0

data_nasabah_baru = pd.DataFrame({
    'Pendapatan_Juta': [input_pendapatan],
    'Riwayat_Baik': [input_riwayat],
    'Ada_Jaminan': [input_jaminan]
})
hasil_prediksi = model.predict(data_nasabah_baru)
print("\n>> HASIL KEPUTUSAN <<")
print(f"Status pengajuan kredit nasabah ini adalah: {hasil_prediksi[0].upper()}")
print("===========================================")