-- Queue Density Monitoring Database Schema for Supabase

-- Create enum for queue status
CREATE TYPE queue_status AS ENUM ('low', 'medium', 'high', 'unknown');

-- Create enum for stall IDs
CREATE TYPE stall_id AS ENUM ('western', 'noodles', 'asian', 'malay', 'indian-deli');

-- Create stalls table to store current queue status
CREATE TABLE stalls (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  stall_id stall_id NOT NULL UNIQUE,
  queue_count INTEGER,
  queue_status queue_status DEFAULT 'unknown',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster lookups by stall_id
CREATE INDEX idx_stalls_stall_id ON stalls(stall_id);

-- Create index for sorting by updated_at
CREATE INDEX idx_stalls_updated_at ON stalls(updated_at DESC);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to auto-update updated_at
CREATE TRIGGER update_stalls_updated_at
  BEFORE UPDATE ON stalls
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Insert initial stall records with unknown status
INSERT INTO stalls (stall_id, queue_count, queue_status) VALUES
  ('western', NULL, 'unknown'),
  ('noodles', NULL, 'unknown'),
  ('asian', NULL, 'unknown'),
  ('malay', NULL, 'unknown'),
  ('indian-deli', NULL, 'unknown');

-- Optional: Create a history table for queue analytics (uncomment if needed)
-- CREATE TABLE queue_history (
--   id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
--   stall_id stall_id NOT NULL,
--   queue_count INTEGER NOT NULL,
--   queue_status queue_status NOT NULL,
--   recorded_at TIMESTAMPTZ DEFAULT NOW()
-- );
-- CREATE INDEX idx_queue_history_stall_id ON queue_history(stall_id);
-- CREATE INDEX idx_queue_history_recorded_at ON queue_history(recorded_at DESC);

-- Note: RLS (Row Level Security) is disabled for now
-- Enable when authentication is implemented
