'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useCurator } from '@/hooks/useCurator'
import { approveChunk, rejectChunk, enrichChunkMetadata } from '@/lib/api/curator'
import { createClient } from '@/lib/supabase/client'
import { ChunkAIMetadata } from '@/types'

async function updateChunkMetadata(chunkId: string, metadata: Partial<ChunkAIMetadata>) {
  const response = await fetch(`/api/chunks/${chunkId}/metadata`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ metadata }),
  })

  if (!response.ok) {
    throw new Error('Failed to update metadata')
  }

  return response.json()
}

interface Props {
  documentId: string
  onComplete?: () => void
}

export default function ChunkReviewer({ documentId, onComplete }: Props) {
  const router = useRouter()
  const {
    document,
    currentChunk,
    currentChunkIndex,
    chunks,
    stats,
    loading,
    error,
    nextChunk,
    previousChunk,
    hasNext,
    hasPrevious,
    progress,
    refreshChunks,
    isComplete,
  } = useCurator(documentId)

  const [curatorNotes, setCuratorNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [enriching, setEnriching] = useState(false)
  const [editingMetadata, setEditingMetadata] = useState(false)
  const [metadataForm, setMetadataForm] = useState<Partial<ChunkAIMetadata>>({})
  const [savingMetadata, setSavingMetadata] = useState(false)

  const handleApprove = async () => {
    if (!currentChunk) return

    setSubmitting(true)
    try {
      const supabase = createClient()
      const {
        data: { user },
      } = await supabase.auth.getUser()

      if (!user) {
        alert('You must be logged in')
        return
      }

      await approveChunk(currentChunk.id, curatorNotes, user.id)
      setCuratorNotes('')
      await refreshChunks()

      // Auto-advance if not last chunk
      if (hasNext && chunks.length > 1) {
        setTimeout(() => nextChunk(), 300)
      }
    } catch (error) {
      console.error('Approve failed:', error)
      alert('Failed to approve chunk. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  const handleReject = async () => {
    if (!currentChunk) return

    if (!confirm('Are you sure you want to reject this chunk?')) {
      return
    }

    setSubmitting(true)
    try {
      const supabase = createClient()
      const {
        data: { user },
      } = await supabase.auth.getUser()

      if (!user) {
        alert('You must be logged in')
        return
      }

      await rejectChunk(currentChunk.id, user.id)
      setCuratorNotes('')
      await refreshChunks()

      // Auto-advance if not last chunk
      if (hasNext && chunks.length > 1) {
        setTimeout(() => nextChunk(), 300)
      }
    } catch (error) {
      console.error('Reject failed:', error)
      alert('Failed to reject chunk. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  const handleEnrichMetadata = async () => {
    if (!currentChunk || !document) return

    setEnriching(true)
    try {
      await enrichChunkMetadata(currentChunk.id, currentChunk.chunk_text, document.doc_type)
      await refreshChunks()
    } catch (error) {
      console.error('Enrich failed:', error)
      alert('Failed to enrich metadata. Please try again.')
    } finally {
      setEnriching(false)
    }
  }

  const handleEditMetadata = () => {
    setMetadataForm(metadata || {})
    setEditingMetadata(true)
  }

  const handleCancelEditMetadata = () => {
    setEditingMetadata(false)
    setMetadataForm({})
  }

  const handleSaveMetadata = async () => {
    if (!currentChunk) return

    setSavingMetadata(true)
    try {
      await updateChunkMetadata(currentChunk.id, metadataForm)
      await refreshChunks()
      setEditingMetadata(false)
      setMetadataForm({})
    } catch (error) {
      console.error('Save metadata failed:', error)
      alert('Failed to save metadata. Please try again.')
    } finally {
      setSavingMetadata(false)
    }
  }

  const updateMetadataField = (field: keyof ChunkAIMetadata, value: unknown) => {
    setMetadataForm(prev => ({ ...prev, [field]: value }))
  }

  const addTag = (tag: string) => {
    if (!tag.trim()) return
    const currentTags = metadataForm.tags || []
    if (!currentTags.includes(tag.trim())) {
      updateMetadataField('tags', [...currentTags, tag.trim()])
    }
  }

  const removeTag = (tagToRemove: string) => {
    const currentTags = metadataForm.tags || []
    updateMetadataField('tags', currentTags.filter(tag => tag !== tagToRemove))
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading chunks...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-red-600 font-medium mb-2">Error loading chunks</div>
        <p className="text-sm text-gray-600">{error}</p>
      </div>
    )
  }

  if (isComplete || chunks.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">🎉</div>
        <h2 className="text-2xl font-bold text-green-600 mb-4">All chunks reviewed!</h2>
        <p className="text-gray-600 mb-6">Great work! This document is complete.</p>
        {document && (
          <div className="bg-gray-50 rounded-lg p-4 max-w-md mx-auto">
            <div className="text-sm text-gray-600 space-y-1">
              <p>
                <strong>Document:</strong> {document.original_filename}
              </p>
              <p>
                <strong>Approved:</strong> {stats.approved} chunks
              </p>
              <p>
                <strong>Rejected:</strong> {stats.rejected} chunks
              </p>
            </div>
          </div>
        )}
        <button
          onClick={() => (onComplete ? onComplete() : router.push('/dashboard'))}
          className="mt-6 bg-blue-600 text-white py-2 px-6 rounded-md hover:bg-blue-700 transition-colors"
        >
          Return to Dashboard
        </button>
      </div>
    )
  }

  if (!currentChunk) {
    return <div className="text-center py-12 text-gray-600">No chunk selected</div>
  }

  const metadata: ChunkAIMetadata | null = currentChunk.ai_metadata as ChunkAIMetadata | null

  return (
    <div className="max-w-6xl mx-auto">
      {/* Document Info */}
      {document && (
        <div className="mb-6 bg-gray-50 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-900">{document.original_filename}</h2>
          <p className="text-sm text-gray-600">{document.doc_type.toUpperCase()} Document</p>
        </div>
      )}

      {/* Progress Stats */}
      <div className="mb-6 grid grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
          <div className="text-sm text-gray-500">Total Chunks</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="text-2xl font-bold text-green-600">{stats.approved}</div>
          <div className="text-sm text-gray-500">Approved</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="text-2xl font-bold text-red-600">{stats.rejected}</div>
          <div className="text-sm text-gray-500">Rejected</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="text-2xl font-bold text-blue-600">{stats.pending}</div>
          <div className="text-sm text-gray-500">Pending</div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-600 mb-2">
          <span>Progress</span>
          <span>{Math.round(progress * 100)}% complete</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-blue-600 h-3 rounded-full transition-all duration-300"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between items-center mb-6">
        <button
          onClick={previousChunk}
          disabled={!hasPrevious}
          className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
        >
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Previous
        </button>

        <div className="text-center">
          <span className="font-medium text-gray-900">
            Chunk {currentChunkIndex + 1} of {chunks.length}
          </span>
        </div>

        <button
          onClick={nextChunk}
          disabled={!hasNext}
          className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
        >
          Next
          <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>

      {/* Chunk Content */}
      <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-900">Chunk Content</h3>
        <div className="bg-gray-50 p-4 rounded-md max-h-96 overflow-y-auto border border-gray-200">
          <pre className="whitespace-pre-wrap text-sm font-mono text-gray-800 leading-relaxed">
            {currentChunk.chunk_text}
          </pre>
        </div>
        <div className="mt-2 text-xs text-gray-500">
          {currentChunk.chunk_size || currentChunk.chunk_text.length} characters
        </div>
      </div>

      {/* Metadata */}
      <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Chunk Metadata
            {currentChunk?.metadata_edited && (
              <span className="ml-2 text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded-full">
                Curator Edited
              </span>
            )}
          </h3>
          <div className="flex gap-2">
            {!metadata && !editingMetadata && (
              <button
                onClick={handleEnrichMetadata}
                disabled={enriching}
                className="text-sm px-3 py-1 bg-purple-100 text-purple-700 rounded-md hover:bg-purple-200 disabled:opacity-50"
              >
                {enriching ? 'Enriching...' : '✨ Generate Metadata'}
              </button>
            )}
            {metadata && !editingMetadata && (
              <button
                onClick={handleEditMetadata}
                className="text-sm px-3 py-1 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200"
              >
                ✏️ Edit Metadata
              </button>
            )}
          </div>
        </div>

        {metadata ? (
          <>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">
                  Topic
                </label>
                <div className="bg-blue-50 px-3 py-2 rounded-md text-sm font-medium text-blue-900">
                  {metadata.topic || 'N/A'}
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">
                  Subtopic
                </label>
                <div className="bg-blue-50 px-3 py-2 rounded-md text-sm font-medium text-blue-900">
                  {metadata.subtopic || 'N/A'}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">
                  Relevance Score
                </label>
                <div className="flex items-baseline">
                  <span className="text-3xl font-bold text-gray-900">
                    {metadata.relevance_score?.toFixed(2) || 'N/A'}
                  </span>
                  <span className="ml-2 text-sm text-gray-500">/ 1.0</span>
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">
                  Confidence
                </label>
                <div className="flex items-baseline">
                  <span className="text-3xl font-bold">
                    {currentChunk.confidence_score ? (
                      <>
                        <span className="mr-2">
                          {currentChunk.confidence_score > 0.7
                            ? '🟢'
                            : currentChunk.confidence_score > 0.4
                              ? '🟡'
                              : '🔴'}
                        </span>
                        <span className="text-gray-900">
                          {currentChunk.confidence_score.toFixed(2)}
                        </span>
                      </>
                    ) : (
                      'N/A'
                    )}
                  </span>
                </div>
              </div>
            </div>

            {metadata.use_cases && metadata.use_cases.length > 0 && (
              <div className="mb-4">
                <label className="block text-xs font-medium text-gray-500 mb-2 uppercase">
                  Use Cases
                </label>
                <div className="flex flex-wrap gap-2">
                  {metadata.use_cases.map((uc: string) => (
                    <span
                      key={uc}
                      className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs font-medium"
                    >
                      {uc.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {metadata.key_concepts && metadata.key_concepts.length > 0 && (
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-2 uppercase">
                  Key Concepts
                </label>
                <div className="flex flex-wrap gap-2">
                  {metadata.key_concepts.map((kc: string) => (
                    <span
                      key={kc}
                      className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-xs font-medium"
                    >
                      {kc}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No AI metadata available for this chunk.</p>
            <p className="text-sm mt-2">Click &quot;Generate Metadata&quot; to enrich this chunk.</p>
          </div>
        )}
      </div>

      {/* Curator Notes */}
      <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6 mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Curator Notes (Optional)
        </label>
        <textarea
          value={curatorNotes}
          onChange={(e) => setCuratorNotes(e.target.value)}
          placeholder="Add any additional context, corrections, or observations..."
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
        />
        <p className="mt-2 text-xs text-gray-500">
          These notes will be stored with the approved chunk and visible to users querying the
          knowledge base.
        </p>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={handleApprove}
          disabled={submitting}
          className="flex-1 bg-green-600 text-white py-4 px-6 rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-semibold text-lg transition-colors shadow-sm"
        >
          {submitting ? 'Processing...' : '✅ Approve'}
        </button>
        <button
          onClick={handleReject}
          disabled={submitting}
          className="flex-1 bg-red-600 text-white py-4 px-6 rounded-md hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-semibold text-lg transition-colors shadow-sm"
        >
          {submitting ? 'Processing...' : '❌ Reject'}
        </button>
      </div>

      <p className="text-sm text-gray-500 mt-4 text-center">
        💡 <strong>Tip:</strong> Review metadata carefully. Your notes will help Members understand
        the context and application of this content.
      </p>
    </div>
  )
}
