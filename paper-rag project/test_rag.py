import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.services.rag_service import process_pdf

pdf_path = r'backend\uploads\1512.03385v1.pdf'
print(f"Processing {pdf_path}...")
try:
    process_pdf(pdf_path)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
