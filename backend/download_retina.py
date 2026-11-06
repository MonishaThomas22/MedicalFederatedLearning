import kagglehub

path = kagglehub.competition_download(
    "aptos2019-blindness-detection",
    output_dir="data/hospital3_retina"
)

print("Download completed!")
print("Dataset location:", path)