'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import type { Stall, QueueStatus } from '@/lib/types';

// Mock data - will be replaced with real data from backend/database later
const stalls: Stall[] = [
  {
    id: 'western',
    name: 'Western',
    queueCount: 8,
    queueStatus: 'medium',
    lastUpdated: new Date(),
  },
  {
    id: 'noodles',
    name: 'Noodles',
    queueCount: null,
    queueStatus: 'unknown',
    lastUpdated: null,
  },
  {
    id: 'asian',
    name: 'Asian',
    queueCount: null,
    queueStatus: 'unknown',
    lastUpdated: null,
  },
  {
    id: 'malay',
    name: 'Malay',
    queueCount: null,
    queueStatus: 'unknown',
    lastUpdated: null,
  },
  {
    id: 'indian-deli',
    name: 'Indian/Deli',
    queueCount: null,
    queueStatus: 'unknown',
    lastUpdated: null,
  },
];

function getStatusColor(status: QueueStatus): string {
  switch (status) {
    case 'low':
      return 'text-green-400';
    case 'medium':
      return 'text-yellow-400';
    case 'high':
      return 'text-red-400';
    default:
      return 'text-gray-500';
  }
}

function getStatusBadge(status: QueueStatus): string {
  switch (status) {
    case 'low':
      return 'bg-green-900/50 text-green-400 border-green-700';
    case 'medium':
      return 'bg-yellow-900/50 text-yellow-400 border-yellow-700';
    case 'high':
      return 'bg-red-900/50 text-red-400 border-red-700';
    default:
      return 'bg-gray-800 text-gray-500 border-gray-700';
  }
}

function getStatusLabel(status: QueueStatus): string {
  switch (status) {
    case 'low':
      return 'Low Queue';
    case 'medium':
      return 'Medium Queue';
    case 'high':
      return 'High Queue';
    default:
      return 'No Data';
  }
}

export default function Home() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white">Dining Hall Queue Status</h1>
              <p className="text-gray-400 mt-1">Real-time queue monitoring for campus dining</p>
            </div>
            <Link
              href="/admin-upload"
              className="px-4 py-2 bg-blue-600 text-white rounded-md font-medium
                hover:bg-blue-700 transition-colors text-sm"
            >
              Admin Upload
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-white mb-2">Current Queue Status</h2>
          <p className="text-gray-400 text-sm">
            Last updated: {mounted ? new Date().toLocaleString() : '...'}
          </p>
        </div>

        {/* Stalls Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {stalls.map((stall) => (
            <div
              key={stall.id}
              className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700
                hover:border-blue-600 transition-colors"
            >
              {/* Stall Name */}
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">{stall.name}</h3>
                <span
                  className={`px-3 py-1 text-xs font-semibold rounded-full border ${getStatusBadge(
                    stall.queueStatus
                  )}`}
                >
                  {getStatusLabel(stall.queueStatus)}
                </span>
              </div>

              {/* Queue Count */}
              <div className="mb-4">
                {stall.queueCount !== null ? (
                  <div className="flex items-baseline gap-2">
                    <span className={`text-4xl font-bold ${getStatusColor(stall.queueStatus)}`}>
                      {stall.queueCount}
                    </span>
                    <span className="text-gray-400 text-sm">people in queue</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-gray-500">
                    <svg
                      className="w-6 h-6"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <span className="text-sm">Queue data unavailable</span>
                  </div>
                )}
              </div>

              {/* Last Updated */}
              <div className="text-xs text-gray-500">
                {stall.lastUpdated ? (
                  <>Updated {stall.lastUpdated.toLocaleTimeString()}</>
                ) : (
                  <>No recent updates</>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Info Banner */}
        <div className="mt-8 bg-blue-900/30 border border-blue-700 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div>
              <h4 className="text-sm font-semibold text-blue-400 mb-1">
                Currently in Beta Testing
              </h4>
              <p className="text-sm text-gray-300">
                Queue monitoring is currently active for the Western stall only. Additional stalls will be added progressively.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
