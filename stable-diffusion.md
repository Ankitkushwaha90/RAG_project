---
license: openrail
datasets:
- fka/awesome-chatgpt-prompts
language:
- en
metrics:
- accuracy
base_model:
- black-forest-labs/FLUX.1-Kontext-dev
new_version: moonshotai/Kimi-K2-Instruct
pipeline_tag: text-to-audio
library_name: adapter-transformers
tags:
- finance
---
# 🚀 Stable Diffusion WebUI - Easy Setup Guide (Windows)

This guide helps you set up **Stable Diffusion WebUI by AUTOMATIC1111** along with **GFPGAN** face restoration on a Windows machine using Git Bash.

---

## ✅ Step-by-Step Installation Instructions

### 🔹 Step 1: Create Required Accounts
Before proceeding, create and log into your accounts on:
- [GitHub](https://github.com/)
- [Hugging Face](https://huggingface.co/)

You’ll need a Hugging Face account to access model files like `stable-diffusion-v1-4`.

---

### 🔹 Step 2: Install Git Bash
Download and install Git for Windows:

🔗 [https://gitforwindows.org/](https://gitforwindows.org/)

---

### 🔹 Step 3: Create a Working Folder
Create a new folder in your **C drive** and name it:
```bash
C:\Ai
```
Copy the full path (right-click the folder > "Copy as path").


### 🔹 Step 4: Open Git Bash and Navigate
Launch **Git Bash**, and run this command to move into the `Ai` folder:

```bash
cd /c/Ai
```
(If your path has spaces or is different, enclose it in double quotes.)

### 🔹 Step 5: Clone Stable Diffusion WebUI Repo
Download AUTOMATIC1111’s WebUI from GitHub:

```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
```
### 🔹 Step 6: Download GFPGAN (Face Restoration Model)
Clone the GFPGAN repository:

```bash
git clone https://github.com/TencentARC/GFPGAN
```

### 🔹 Step 7: Install Python 3.10.6
Download and install the exact version:

🔗 Python 3.10.6 Download

📌 Ensure you check “Add Python to PATH” during installation.

### 🔹 Step 8: Download Stable Diffusion v1-4 Checkpoint
Visit the model page on Hugging Face:
🔗 https://huggingface.co/CompVis/stable-diffusion-v1-4

- Click “Access repository” and accept terms.

- Download the v1-4-pruned-emaonly.ckpt file.

### 🔹 Step 9: Organize Files
Move the GFPGAN folder into the stable-diffusion-webui directory:

```makefile
C:\Ai\stable-diffusion-webui\GFPGAN
```

Move the v1-4-pruned-emaonly.ckpt file into:

```pgsql
C:\Ai\stable-diffusion-webui\models\Stable-diffusion
```
### 🔹 Step 10: Launch the WebUI
Go to your stable-diffusion-webui folder:

```bash
cd stable-diffusion-webui
```
Run the WebUI:

```bash
./webui.bat
```
Wait for it to complete setup and load. Once done, you’ll see a local URL like:

```nginx
Running on local URL: http://127.0.0.1:7860
```
Open that URL in any browser. You're ready to generate AI images!

### 🎉 You're Done!
You now have Stable Diffusion WebUI running locally with support for:

- GFPGAN (face enhancement)

- v1-4 Stable Diffusion model

### 🛠️ Tips
To upgrade later, just run:

```bash
git pull
```
For additional models, place .ckpt or .safetensors files inside models/Stable-diffusion/.

### 📎 Resources
AUTOMATIC1111 WebUI

- GFPGAN

- Stable Diffusion v1-4
