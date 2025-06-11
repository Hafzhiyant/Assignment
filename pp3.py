from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ganti path di bawah sesuai lokasi file PDF-mu
file_paths = [
    r"C:\Users\ediso\Downloads\paper1.pdf",
    r"C:\Users\ediso\Downloads\paper2.pdf",
    r"C:\Users\ediso\Downloads\paper3.pdf"
]

# Ambil teks abstrak dari halaman pertama tiap PDF
documents = []
for path in file_paths:
    reader = PdfReader(path)
    page = reader.pages[0]  # Ambil halaman pertama
    text = page.extract_text()
    documents.append(text.strip())

# Hitung TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Hitung cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Tampilkan hasil
print("Cosine Similarity Matrix (dokumen 1-3):")
for i in range(len(similarity_matrix)):
    for j in range(len(similarity_matrix)):
        print(f"Similarity(doc{i+1}, doc{j+1}) = {similarity_matrix[i][j]:.4f}")