import json
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 1. Koneksi ke workspace Roboflow Anda
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="XoKyoGbEmr2MNaoq4HKg"  # API key Anda
).configure(InferenceConfiguration(
    api_key_transport="header"
))

# 2. Jalankan workflow pada gambar
# Ganti "path/ke/gambar/cabai1.jpg" dengan lokasi file foto Anda yang sebenarnya
image_path = "cabai1.jpg" 

try:
    result = client.run_workflow(
        workspace_name="ahmad-sulthoni",
        workflow_id="general-segmentation-api-3",
        images={
            "image": image_path
        },
        parameters={
            "classes": "Bercak Daun Serkospora, Buah Cabai Sehat, Busuk Buah Antraknosa, Daun Cabai Sehat, Virus Kuning"
        },
        use_cache=True
    )

    # 3. Tampilkan hasil JSON mentah secara rapi
    print("--- Hasil Deteksi (JSON) ---")
    print(json.dumps(result, indent=2))

except Exception as e:
    print(f"Terjadi kesalahan: {e}")