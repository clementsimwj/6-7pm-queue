'use client';

import { useState, useRef } from 'react';
import type { StallId, QueueStatus } from '@/lib/types';
import Navbar from '../components/Navbar';

// Helper function to determine queue status based on count
function getQueueStatus(count: number): QueueStatus {
  if (count <= 3) return 'low';
  if (count <= 7) return 'medium';
  return 'high';
}

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [selectedStall, setSelectedStall] = useState<StallId>('western');
  const [queueCount, setQueueCount] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    setSelectedFile(file);
    setError(null);

    // Create preview URL for original image
    const previewUrl = URL.createObjectURL(file);
    setOriginalPreview(previewUrl);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Step 1: Get CV analysis from Python backend
      const response = await fetch('/api/cv/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `Request failed with status ${response.status}`);
      }

      // Get the annotated image as a blob
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      
      setAnnotatedImage(imageUrl);
      setLastUpdated(new Date());

      // TODO: Parse queue count from Python response
      // For now, using a mock value - this will be replaced when Python backend
      // returns JSON with queue count
      const mockQueueCount = 8; // This should come from Python API
      const queueStatus = getQueueStatus(mockQueueCount);
      setQueueCount(mockQueueCount);

      // Step 2: Update Supabase with the results
      const updateResponse = await fetch('/api/stalls/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stallId: selectedStall,
          queueCount: mockQueueCount,
          queueStatus,
        }),
      });

      if (!updateResponse.ok) {
        console.error('Failed to update database, but image was processed');
        // Don't throw - we still want to show the annotated image
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image');
      console.error('Upload error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setOriginalPreview(null);
    setAnnotatedImage(null);
    setError(null);
    setLastUpdated(null);
    setQueueCount(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Navbar */}
      <Navbar />
      
      <div className="py-8 px-4">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2">Queue Density Analyzer</h1>
          <p className="text-gray-400 mb-8">Upload an image to detect and count people in queue</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Panel: Upload Controls */}
          <div className="bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-white mb-4">Upload Image</h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Select Stall
                </label>
                <select
                  value={selectedStall}
                  onChange={(e) => setSelectedStall(e.target.value as StallId)}
                  className="block w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white
                    focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="western">Western</option>
                  <option value="noodles" disabled>Noodles (Coming Soon)</option>
                  <option value="asian" disabled>Asian (Coming Soon)</option>
                  <option value="malay" disabled>Malay (Coming Soon)</option>
                  <option value="indian-deli" disabled>Indian/Deli (Coming Soon)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Select Image
                </label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="block w-full text-sm text-gray-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-blue-600 file:text-white
                    hover:file:bg-blue-700
                    cursor-pointer"
                />
              </div>

              {originalPreview && (
                <div>
                  <p className="text-sm font-medium text-gray-300 mb-2">Original Image</p>
                  <img
                    src={originalPreview}
                    alt="Original preview"
                    className="w-full rounded-lg border border-gray-700"
                  />
                </div>
              )}

              {error && (
                <div className="bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-md">
                  <p className="text-sm">{error}</p>
                </div>
              )}

              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={!selectedFile || isProcessing}
                  className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md font-medium
                    hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed
                    transition-colors"
                >
                  {isProcessing ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </span>
                  ) : (
                    'Analyze Queue'
                  )}
                </button>

                {(selectedFile || annotatedImage) && (
                  <button
                    type="button"
                    onClick={handleClear}
                    disabled={isProcessing}
                    className="px-4 py-2 border border-gray-600 rounded-md text-gray-300
                      hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed
                      transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </form>

            <div className="mt-6 p-4 bg-gray-700 rounded-md">
              <h3 className="text-sm font-semibold text-blue-400 mb-2">Legend</h3>
              <div className="space-y-1 text-sm text-gray-300">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-green-500 rounded"></div>
                  <span>Green boxes: People counted in queue</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-red-500 rounded"></div>
                  <span>Red boxes: People filtered out (too distant/small)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel: Results */}
          <div className="bg-gray-800 rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Analysis Result</h2>
              {lastUpdated && (
                <p className="text-sm text-gray-400">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </p>
              )}
            </div>

            {annotatedImage ? (
              <div>
                {queueCount !== null && (
                  <div className="mb-4 p-4 bg-blue-900/30 border border-blue-700 rounded-md">
                    <p className="text-sm text-gray-300">
                      Queue Count: <span className="text-2xl font-bold text-blue-400">{queueCount}</span> people
                    </p>
                  </div>
                )}
                <img
                  src={annotatedImage}
                  alt="Annotated result"
                  className="w-full rounded-lg border border-gray-700"
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-96 bg-black rounded-lg border-2 border-dashed border-gray-700">
                <div className="text-center text-gray-500">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <p className="mt-2">Upload and analyze an image to see results</p>
                </div>
              </div>
            )}
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}
