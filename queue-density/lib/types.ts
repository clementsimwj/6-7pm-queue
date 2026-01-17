/**
 * Shared TypeScript types for the Queue Density application
 */

export type QueueStatus = 'low' | 'medium' | 'high' | 'unknown';

export type StallId = 'western' | 'noodles' | 'asian' | 'malay' | 'indian-deli';

export interface Stall {
  id: StallId;
  name: string;
  queueCount: number | null;
  queueStatus: QueueStatus;
  lastUpdated: Date | null;
}

/**
 * Database schema types (for Supabase integration)
 * Note: Images are NOT stored - only the analysis results
 */
export interface StallRecord {
  id: string;
  stall_id: StallId;
  queue_count: number;
  queue_status: QueueStatus;
  created_at: string;
  updated_at: string;
}

/**
 * API response from CV analysis endpoint
 * Note: annotatedImage is returned as blob/stream for display only, not stored
 */
export interface CVAnalysisResponse {
  queueCount: number;
  queueStatus: QueueStatus;
}
