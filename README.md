CvScout - replacement project files

Files:
- app.py: Streamlit application
- requirements.txt: required Python packages

Run locally from the project folder:
1. Open the folder in VS Code.
2. Open Terminal in VS Code.
3. Run: pip install -r requirements.txt
4. Run: streamlit run app.py

For deployment:
1. Replace app.py and requirements.txt in your existing CvScout GitHub repository.
2. Commit and push the changes.
3. Streamlit Community Cloud will redeploy the app from the repository.

Notes:
- The first run downloads the free pretrained all-MiniLM-L6-v2 Sentence Transformer model.
- Use text-based PDFs or DOCX files. Scanned/image-only PDFs without embedded text need OCR, which is not included here.
- Relevance scores are ranking estimates, not measured accuracy or hiring decisions. Review borderline candidates manually.
