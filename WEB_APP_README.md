# Web Application Guide

This project includes a Streamlit web application for generating cover letters using RAG + LoRA.

## Streamlit App

**Easy to use** - Perfect for demos and quick deployment.

### Setup
```bash
pip install streamlit
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app_streamlit.py
```

Or use the convenience script:
```bash
./run_streamlit.sh
```

The app will open in your browser at `http://localhost:8501`

### Features
- ✅ Simple, clean interface
- ✅ Two-column layout (resume + job description)
- ✅ Built-in file download
- ✅ Loading indicators
- ✅ No HTML/CSS/JS needed
- ✅ Great for demos and prototyping

---

## Deployment Options

### Streamlit Cloud (Free & Recommended)
1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo
4. Deploy!

### Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t coverletter-rag .
docker run -p 8501:8501 coverletter-rag
```

### Other Platforms
- **Heroku**: Use the Dockerfile approach
- **AWS EC2**: Run `streamlit run app_streamlit.py --server.port=8501 --server.address=0.0.0.0`
- **Google Cloud Run**: Use Dockerfile
- **Azure**: Use Dockerfile

---

## Troubleshooting

### Model Not Found Error
Make sure you've:
1. Run `python scripts/build_original_dataset.py`
2. Run `python scripts/build_embeddings.py`
3. Run `python src/train.py --config_name rag_lora`

### Slow First Generation
The first generation loads the model into memory (~1-2 minutes). Subsequent generations are faster.

### Memory Issues
If you run out of memory:
- Use CPU instead of GPU (already configured)
- Reduce batch size in config
- Consider using a smaller base model

### Port Already in Use
If port 8501 is taken, use:
```bash
streamlit run app_streamlit.py --server.port 8502
```

---

## Customization

Edit `app_streamlit.py` to customize:
- Colors and styling (CSS in `st.markdown()`)
- Layout (column arrangements)
- Additional features (file upload, history, etc.)
- Sidebar content

---

## Usage Tips

1. **First Run**: The first generation takes longer as the model loads
2. **Input Quality**: More detailed resume and job descriptions produce better results
3. **Review**: Always review and edit the generated cover letter before using
4. **Personalization**: The model extracts key bullets from your resume automatically
