# import pandas as pd

# df = pd.read_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_train.csv", sep="|", header=None, names=["audio","text"])
# df["audio"] = "wavs/" + df["audio"]
# df["speaker"] = "speaker_1"
# df.to_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_fixed.csv", sep="|", header=False, index=False)


# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load your fixed metadata file
# df = pd.read_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_fixed.csv", sep="|", header=None, names=["audio","text","speaker"])

# # Split: 95% train, 5% eval
# train_df, eval_df = train_test_split(df, test_size=0.05, random_state=42)

# # Save both files without header
# train_df.to_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_train.csv", sep="|", header=False, index=False)
# eval_df.to_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_eval.csv", sep="|", header=False, index=False)

# print("✅ Split completed: metadata_train.csv & metadata_eval.csv created.")


import pandas as pd

# === Add header to metadata_train.csv ===
train_df = pd.read_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_train.csv", sep="|", header=None, names=["audio_file", "text", "speaker"])
train_df.to_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_train.csv", sep="|", index=False)

# === Add header to metadata_eval.csv ===
eval_df = pd.read_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_eval.csv", sep="|", header=None, names=["audio_file", "text", "speaker"])
eval_df.to_csv(r"D:\UMER\xtts-v2-ft\XTTSv2-Finetuning-for-New-Languages\datasets\metadata_eval.csv", sep="|", index=False)

print("✅ Headers added to both train and eval metadata files.")
