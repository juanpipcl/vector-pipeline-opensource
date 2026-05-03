# 🚀 Serverless Vector Pipeline (Open Source)

Este proyecto implementa un pipeline de datos para convertir documentos no estructurados (PDFs) en vectores de alta dimensionalidad, almacenándolos en una base de datos vectorial para búsqueda semántica.

**Enfoque:** Freelance / Cost-Zero (Totalmente gratuito usando capas Open Source y Cloud Free-Tiers).

## 🛠️ Arquitectura
- **Lenguaje:** Python 3.10
- **Orquestación:** GitHub Actions (CI/CD Pipeline)
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`) - *Local & Free*
- **Vector Database:** Pinecone (Serverless Tier)
- **Framework:** LangChain

## 📦 Instalación Local
1. Clonar el repositorio.
2. Crear entorno virtual: `python -m venv venv`.
3. Activar: `source venv/bin/activate` (o `.\venv\Scripts\activate` en Windows).
4. Instalar: `pip install -r requirements.txt`.

## ☁️ Despliegue en la Nube
El pipeline está configurado para ejecutarse automáticamente mediante **GitHub Actions** cada vez que se detecta un nuevo archivo en la carpeta `/data`.

### Requisitos para la Nube:
Configurar en los **Secrets** de GitHub:
- `PINECONE_API_KEY`: Tu llave de Pinecone.

## 📄 Licencia
Este proyecto es Open Source bajo la licencia MIT.