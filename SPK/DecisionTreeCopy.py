import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# =====Langkah Pertama=====
# buat sebuah variabel yang digunakan untuk menampung fungsi pembaca file csv kamu
# selanjutnya buat variiabel yang digunakan untuk menampung fungsi pemisah kolom
# terakhir, buat variabel yang digunakan untuk menampung salah satu kolom yang dihapus tadi
df = pd.read_csv('Dataset_Kelayakan_Kredit.csv')

x = df.drop(columns=['Status'])
y = df['Status']

# =====Langkah Kedua=====
# buat variabel untuk menampung fungsi decisiontree
# latih model tersebut dengan variabel x,y yang telah kita buat
## variabel random_state=42 digunakan untuk memastikan hasil pohon tetap konsisten saat program dijalankan
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

model.fit(x, y)

print("Model berhasi di Training. Kini model sudah siap untuk digunakan")


print("Sedang menggambar pohon keputusan")
# buat ukuran kanvas
plt.figure(figsize=(14,8))

# buat fungsi untuk menggambar pohon
tree.plot_tree(model,
               feature_names=['Pendapatan_Juta', 'Riwayat_Baik', 'Ada_Jaminan'],
               class_names=model.classes_,
               filled=True,
               rounded=True,
               fontsize=10)

plt.title("Visualisasi Pohon Keputusan Kelayakan Kredit")
plt.savefig("Pohon_keputusan.png")
print("Berhasil, gambar pohon keputusan telah disimpan sebagai 'Pohon_Keputusan.png'.\n")
#plt.show()

# tahap input
print("== SISTEM KEPUTUSAN UNTUK MENENTUKAN KELAYAKAN KREDIT")
input_pendapatan = float(input('Masukkan pendapatan calon nasabah: '))
input_riwayat = int(input("Masukkan riwayat kredit calon nasabah"))
input_jaminan = int(input("Masukkan jaminan yang dimiliki oleh calon nasabah: "))

data_nasabah_baru = pd.DataFrame({
    'Pendapatan_Juta': [input_pendapatan],
    'Riwayat_Baik': [input_riwayat],
    'Ada_Jaminan': [input_jaminan]
})
hasil_prediksi = model.predict(data_nasabah_baru)
print("\n>> HASIL KEPUTUSAN <<")
print(f"Status pengajuan kredit nasabah ini adalah: {hasil_prediksi[0].upper()}")
print("===========================================")