'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';
import type { Stall, QueueStatus, StallId } from '@/lib/types';
import Navbar from './components/Navbar';

// Stall display names mapping
const STALL_NAMES: Record<StallId, string> = {
  western: 'Western',
  noodles: 'Noodles',
  asian: 'Asian',
  malay: 'Malay',
  'indian-deli': 'Indian/Deli',
};

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
  const [stalls, setStalls] = useState<Stall[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    fetchStalls();

    // Set up real-time subscription
    const channel = supabase
      .channel('stalls-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'stalls',
        },
        () => {
          // Refetch data when any change occurs
          fetchStalls();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const fetchStalls = async () => {
    try {
      const { data, error } = await supabase
        .from('stalls')
        .select('*')
        .order('stall_id', { ascending: true });

      if (error) throw error;

      // Transform database records to Stall type
      const transformedStalls: Stall[] = (data || []).map((record) => ({
        id: record.stall_id,
        name: STALL_NAMES[record.stall_id],
        queueCount: record.queue_count,
        queueStatus: record.queue_status,
        lastUpdated: record.updated_at ? new Date(record.updated_at) : null,
      }));

      setStalls(transformedStalls);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching stalls:', err);
      setError(err instanceof Error ? err.message : 'Failed to load stalls');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center gap-4 mb-8">
          <Image
            src="/logo_main.png"
            alt="6-7pm Queue Logo"
            width={60}
            height={60}
            className="object-contain"
          />
          <div>
            <h1 className="text-3xl font-bold text-white">6-7pm Queue Monitor</h1>
            <p className="text-gray-400 mt-1">Real-time queue monitoring for campus dining</p>
          </div>
        </div>

        <div className="mb-6">
          <h2 className="text-xl font-semibold text-white mb-2">Current Queue Status</h2>
          <p className="text-gray-400 text-sm">
            Last updated: {mounted ? new Date().toLocaleString() : '...'}
          </p>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <svg
                className="animate-spin h-8 w-8 text-blue-500 mx-auto mb-3"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              <p className="text-gray-400">Loading stalls...</p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-md mb-6">
            <p className="text-sm">Error: {error}</p>
          </div>
        )}

        {/* Stalls Grid */}
        {!loading && !error && (
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
        )}

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
                Currently in Proof of Concept Phase
              </h4>
              <p className="text-sm text-gray-300">
                Queue monitoring is currently active for the Western stall only.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
