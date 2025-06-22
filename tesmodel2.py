import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models # Pastikan ini diimpor untuk VGG19
import numpy as np

# --- Konfigurasi Aplikasi Flask ---
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Batas ukuran file 16MB

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Konfigurasi Device (GPU/CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device untuk inferensi: {device}")

# === Arsitektur Model Generator (SRResNet) ===
# Salin kelas ResidualBlock dan SRResNetGenerator dari kode pelatihan Anda
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x) 
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class SRResNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16, base_channels=64):
        super(SRResNetGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()

        self.residual_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(n_residual_blocks)])

        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(base_channels)

        # Upsampling blocks (2x upscale per block, total 4x)
        self.upsample1 = self._upsample_block(base_channels)
        self.upsample2 = self._upsample_block(base_channels)

        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=9, padding=4)

    def _upsample_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2), # PixelShuffle performs 2x upscale
            nn.PReLU()
        )

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.residual_blocks(out1)
        out = self.bn(self.conv2(out))
        out = out + out1 # Global skip connection

        out = self.upsample1(out) # First 2x upscale
        out = self.upsample2(out) # Second 2x upscale (total 4x)

        out = self.conv3(out)
        return torch.sigmoid(out) # Pastikan output dalam rentang [0, 1]

# --- Muat Model yang Sudah Dilatih ---
# Nama file model Anda
MODEL_PATH = "srresnet_perceptual_rgb_new.pth" 
# Faktor downscale yang digunakan saat pelatihan (misal: 4x)
DOWNSCALE_FACTOR = 4 
# Ukuran gambar HR yang digunakan saat pelatihan (misal: 256x256)
TARGET_HR_SIZE = (500, 500) 

# Inisialisasi model dan pindahkan ke device
try:
    model = SRResNetGenerator(
        in_channels=3, out_channels=3, n_residual_blocks=16, base_channels=64
    ).to(device)
    # Muat state dictionary model ke device yang sesuai
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Penting: Atur model ke mode evaluasi untuk inferensi
    print(f"Model '{MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Pastikan file model ada dan arsitektur kelas SRResNetGenerator benar. {e}")
    model = None # Set model ke None agar aplikasi bisa tahu jika ada error

# --- Fungsi Bantu untuk Validasi File ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Fungsi Inti untuk Inferensi Super Resolution ---
# MODIFIKASI: Hanya kembalikan SR output dan dimensi input/output
def perform_super_resolution(input_image_path, sr_model, downscale_factor, target_hr_size):
    if sr_model is None:
        raise ValueError("Model belum dimuat atau ada kesalahan saat memuat model.")

    # Buka gambar input
    original_image = Image.open(input_image_path).convert('RGB')
    
    # Simpan dimensi asli dari gambar yang diupload (sebelum di-resize untuk pemrosesan)
    # Ini akan menjadi original_lr_dimensions di tampilan
    original_uploaded_dims = original_image.size # (width, height)

    # Sesuaikan ukuran gambar input ke ukuran target HR (500x500)
    # Ini adalah ukuran basis untuk membuat LR input ke model
    hr_base_image_for_processing = original_image.resize(target_hr_size, Image.BICUBIC)

    # Buat versi LR kecil sebagai input ke model
    lr_width_model_input = target_hr_size[1] // downscale_factor
    lr_height_model_input = target_hr_size[0] // downscale_factor
    lr_size_model_input = (lr_width_model_input, lr_height_model_input)
    lr_image_small_for_model = hr_base_image_for_processing.resize(lr_size_model_input, Image.BICUBIC)
    
    # Transformasi input ke tensor PyTorch
    transform_to_tensor = transforms.ToTensor()
    lr_image_small_tensor = transform_to_tensor(lr_image_small_for_model).unsqueeze(0).to(device) # unsqueeze(0) untuk batch dimension

    # Lakukan inferensi (prediksi) dengan model
    with torch.no_grad(): # Matikan perhitungan gradien untuk inferensi
        sr_output_tensor = sr_model(lr_image_small_tensor)

    # Konversi output tensor kembali ke gambar PIL
    sr_output_image_np = sr_output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_output_image_np = np.clip(sr_output_image_np, 0, 1) # Pastikan nilai piksel di antara 0 dan 1
    sr_output_image_pil = Image.fromarray((sr_output_image_np * 255).astype(np.uint8))
    
    # MODIFIKASI PENGEMBALIAN: Hanya kembalikan gambar SR dan dimensi yang relevan
    return sr_output_image_pil, original_uploaded_dims, sr_output_image_pil.size 

# --- Rute Aplikasi Flask ---
# Rute utama untuk menampilkan form upload dan hasil
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="Tidak ada file yang dipilih.")
        
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error="Nama file kosong. Pilih file gambar.")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if model is None:
                return render_template('index.html', error="Error: Model gagal dimuat. Cek log server.")
            
            try:
                # MODIFIKASI PANGGILAN FUNGSI: Hanya ambil data yang diperlukan
                sr_image_pil, original_input_dims, sr_dims = perform_super_resolution(
                    filepath, model, DOWNSCALE_FACTOR, TARGET_HR_SIZE
                )
                
                # Simpan gambar hasil SR
                result_filename = "sr_" + filename
                result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                sr_image_pil.save(result_filepath)
                
                # MODIFIKASI PENGIRIMAN KE TEMPLATE: Hanya kirim data yang diperlukan
                return render_template('index.html', 
                                       original_image=url_for('uploaded_file', filename=filename), # Gambar yang diupload pengguna
                                       original_lr_dimensions=f"{original_input_dims[0]}x{original_input_dims[1]}", # Dimensi gambar yang diupload
                                       
                                       super_resolved_image=url_for('result_file', filename=result_filename), # Gambar hasil SR
                                       sr_dimensions=f"{sr_dims[0]}x{sr_dims[1]}")
            except Exception as e:
                # Tangani error selama proses SR
                print(f"Error during Super Resolution: {e}")
                return render_template('index.html', error=f"Terjadi kesalahan saat memproses gambar: {e}")
        else:
            # Jika file tidak diizinkan
            return render_template('index.html', error="Tipe file tidak diizinkan. Hanya PNG, JPG, JPEG.")
    
    # Jika metode GET (saat pertama kali membuka halaman)
    return render_template('index.html')

# Rute untuk menyajikan gambar yang diupload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Rute untuk menyajikan gambar hasil (SR)
@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)
    
if __name__ == '__main__':
    app.run(debug=True) # debug=True untuk pengembangan, matikan saat produksi