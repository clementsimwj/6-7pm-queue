import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

/**
 * Proxy endpoint that forwards image uploads to the FastAPI CV service
 * and returns the annotated image result.
 * 
 * This proxy approach:
 * - Avoids CORS issues
 * - Keeps the CV API URL server-side and private
 * - Maintains a clean separation between Next.js UI and Python CV service
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file');

    if (!file || !(file instanceof File)) {
      return NextResponse.json(
        { error: 'No file provided or invalid file format' },
        { status: 400 }
      );
    }

    // Validate file type
    if (!file.type.startsWith('image/')) {
      return NextResponse.json(
        { error: 'File must be an image' },
        { status: 400 }
      );
    }

    const cvApiUrl = process.env.CV_API_URL;
    if (!cvApiUrl) {
      console.error('CV_API_URL environment variable is not set');
      return NextResponse.json(
        { error: 'CV API configuration missing' },
        { status: 500 }
      );
    }

    // Create new FormData to forward to FastAPI
    const forwardFormData = new FormData();
    forwardFormData.append('file', file);

    // Forward the request to FastAPI
    const response = await fetch(`${cvApiUrl}/count_debug`, {
      method: 'POST',
      body: forwardFormData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('FastAPI error:', errorText);
      return NextResponse.json(
        { error: `CV API returned error: ${response.status}` },
        { status: response.status }
      );
    }

    // Get the image buffer from FastAPI
    const imageBuffer = await response.arrayBuffer();

    // Return the image with proper headers
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    console.error('Error in CV proxy:', error);
    return NextResponse.json(
      { error: 'Failed to process image' },
      { status: 500 }
    );
  }
}
