import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import type { StallId, QueueStatus } from '@/lib/types';

export const runtime = 'nodejs';

/**
 * Update stall queue data in Supabase
 * Called after CV analysis is complete
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { stallId, queueCount, queueStatus } = body as {
      stallId: StallId;
      queueCount: number;
      queueStatus: QueueStatus;
    };

    // Validate inputs
    if (!stallId || queueCount === undefined || !queueStatus) {
      return NextResponse.json(
        { error: 'Missing required fields: stallId, queueCount, queueStatus' },
        { status: 400 }
      );
    }

    // Update the stall record in Supabase
    const { data, error } = await supabase
      .from('stalls')
      .update({
        queue_count: queueCount,
        queue_status: queueStatus,
      })
      .eq('stall_id', stallId)
      .select()
      .single();

    if (error) {
      console.error('Supabase error:', error);
      return NextResponse.json(
        { error: 'Failed to update stall data' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      data,
    });
  } catch (error) {
    console.error('Error updating stall:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
