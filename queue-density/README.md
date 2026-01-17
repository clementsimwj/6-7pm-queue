# Queue Density Analyzer

This monorepo contains:
- **Next.js App** (root) - Frontend UI with App Router
- **Python FastAPI CV Service** (`cv-api/`) - Computer vision backend for queue analysis

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.12+

### 1. Set Up Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.local.example .env.local
```

The default configuration points to `http://localhost:8000` for the CV API.

### 2. Run the FastAPI CV Service

Navigate to the `cv-api` directory and start the FastAPI server:

```bash
cd cv-api

# Create and activate virtual environment (first time only)
python -m venv .venv
.venv/Scripts/activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the server
.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
```

The CV API will be available at [http://localhost:8000](http://localhost:8000)

### 3. Run the Next.js Frontend

In a separate terminal, from the project root:

```bash
npm install  # First time only
npm run dev
```

The Next.js app will be available at [http://localhost:3000](http://localhost:3000)

### 4. Access the Upload UI

Open your browser to [http://localhost:3000/admin-upload](http://localhost:3000/admin-upload)

## How It Works

1. User selects an image in the Next.js UI
2. Image is uploaded to Next.js API route (`/api/cv/analyze`)
3. Next.js API proxy forwards the image to FastAPI (`/count_debug` endpoint)
4. FastAPI runs YOLOv8 pose detection to identify people
5. People are classified as "in queue" (green boxes) or "filtered out" (red boxes) based on their height ratio
6. Annotated image is returned through the proxy to the browser

This proxy approach:
- Avoids CORS issues
- Keeps CV API URL private (server-side only)
- Maintains clean separation between frontend and CV backend

## Project Structure

```
queue-density/
├── app/                    # Next.js App Router
│   ├── admin-upload/       # Queue analysis UI page
│   └── api/cv/analyze/     # API proxy to FastAPI
├── cv-api/                 # Python FastAPI service
│   ├── main.py            # CV endpoints
│   ├── requirements.txt   # Python dependencies
│   └── .venv/            # Virtual environment (not committed)
└── .env.local             # Environment variables (not committed)
```

## Legend

When viewing analysis results:
- **Green boxes**: People counted in queue
- **Red boxes**: People filtered out (too distant/small based on height ratio)

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
