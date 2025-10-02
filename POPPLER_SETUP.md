# 📚 Poppler Setup Guide for Windows

## What is Poppler?

Poppler is a PDF rendering library that enables PDF-to-image conversion. It's required for OCR on scanned PDF certificates.

## Current Status

✅ **Image Files (PNG/JPG/JPEG)**: Working perfectly - use these!  
⚠️ **PDF Files**: Limited support without Poppler

## Do You Need Poppler?

**NO, if:**
- You only use image certificates (PNG, JPG, JPEG)
- Your PDFs have selectable text (can be extracted directly)

**YES, if:**
- You need to OCR scanned PDF certificates
- Your PDFs are image-based without text layer

## How to Install Poppler on Windows

### Option 1: Quick Install (Recommended)

1. **Download Poppler for Windows:**
   - Go to: https://github.com/oschwartz10612/poppler-windows/releases/
   - Download the latest release (e.g., `Release-24.08.0-0.zip`)

2. **Extract the ZIP file:**
   - Extract to a permanent location (e.g., `C:\Program Files\poppler`)
   - You should see folders: `Library`, `bin`, etc.

3. **Add to System PATH:**
   
   **Method A - Using Windows Settings:**
   - Press `Win + X` → Select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find and select "Path"
   - Click "Edit"
   - Click "New"
   - Add: `C:\Program Files\poppler\Library\bin` (adjust to your install location)
   - Click "OK" on all dialogs
   
   **Method B - Using PowerShell (Run as Administrator):**
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\poppler\Library\bin", [EnvironmentVariableTarget]::Machine)
   ```

4. **Restart PowerShell/Terminal:**
   - Close all PowerShell windows
   - Open a new PowerShell window
   - Test with: `pdftoppm -v`
   - You should see version information

5. **Update streamlit_app.py:**
   
   Open `streamlit_app.py` and find this section near the top:
   
   ```python
   # POPPLER CONFIGURATION
   POPPLER_PATH = None  # Set this to your Poppler bin path if needed
   ```
   
   Change it to:
   
   ```python
   # POPPLER CONFIGURATION
   POPPLER_PATH = r"C:\Program Files\poppler\Library\bin"  # Your actual path
   ```

6. **Restart Streamlit:**
   - Stop the Streamlit app (Ctrl+C)
   - Run: `streamlit run streamlit_app.py`

### Option 2: Conda Install

If you're using Anaconda/Miniconda:

```bash
conda install -c conda-forge poppler
```

## Verify Installation

### Check if Poppler is in PATH:

```powershell
pdftoppm -v
```

Expected output:
```
pdftoppm version XX.XX.X
Copyright 2005-2024 The Poppler Developers - http://poppler.freedesktop.org
...
```

### Test in Python:

```python
from pdf2image import convert_from_path
# Should not raise an error
```

## Troubleshooting

### "pdftoppm is not recognized"

**Problem:** Poppler bin folder is not in PATH

**Solutions:**
1. Verify the PATH was added correctly
2. Restart your terminal/PowerShell
3. Restart VS Code if running from there
4. Try the absolute path method in `streamlit_app.py`

### "Unable to get page count"

**Problem:** Poppler path is incorrect or not accessible

**Solutions:**
1. Check that `pdftoppm.exe` exists in the bin folder
2. Try setting `POPPLER_PATH` directly in code:
   ```python
   POPPLER_PATH = r"C:\Program Files\poppler\Library\bin"
   ```
3. Use forward slashes: `"C:/Program Files/poppler/Library/bin"`

### PATH not persisting

**Problem:** PATH resets after closing PowerShell

**Solution:** 
- Use "System variables" not "User variables" when adding to PATH
- Run PowerShell as Administrator when setting environment variables

## Alternative: Use Image Files Instead

If Poppler setup is too complex, simply:

1. Convert your PDF certificates to PNG/JPG:
   - Use online tools (e.g., PDF2PNG.com)
   - Use Adobe Acrobat
   - Use Windows "Print to PDF" → "Microsoft Print to PDF"
   - Use screenshot tools (Win + Shift + S)

2. Upload the image files to the app - works perfectly without Poppler!

## Current Workaround

The app is configured to:
- ✅ Work perfectly with image files (PNG/JPG/JPEG)
- ✅ Extract text from PDFs with text layers (no Poppler needed)
- ⚠️ Show warning for scanned PDFs without Poppler
- ✅ Allow manual input as fallback

**You can use the app right now with image files!**

## Need Help?

1. Check that you're using image files (PNG/JPG)
2. Try the sample certificates in `sample_certificates/` folder
3. Use manual input mode if OCR isn't working
4. Verify Poppler installation with `pdftoppm -v`

---

**Bottom Line:** For most users, using PNG/JPG images is easier and works perfectly! 🎉
