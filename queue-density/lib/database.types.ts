export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      stalls: {
        Row: {
          id: string
          stall_id: 'western' | 'noodles' | 'asian' | 'malay' | 'indian-deli'
          queue_count: number | null
          queue_status: 'low' | 'medium' | 'high' | 'unknown'
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          stall_id: 'western' | 'noodles' | 'asian' | 'malay' | 'indian-deli'
          queue_count?: number | null
          queue_status?: 'low' | 'medium' | 'high' | 'unknown'
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          stall_id?: 'western' | 'noodles' | 'asian' | 'malay' | 'indian-deli'
          queue_count?: number | null
          queue_status?: 'low' | 'medium' | 'high' | 'unknown'
          created_at?: string
          updated_at?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      queue_status: 'low' | 'medium' | 'high' | 'unknown'
      stall_id: 'western' | 'noodles' | 'asian' | 'malay' | 'indian-deli'
    }
  }
}
