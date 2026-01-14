import { notFound } from 'next/navigation'
import ChunkReviewer from '@/components/curator/ChunkReviewer'
import { createClient } from '@/lib/supabase/server'

async function getDocument(docId: string) {
  const supabase = await createClient()
  const { data, error } = await supabase
    .from('documents')
    .select(`
      *,
      profiles!documents_uploaded_by_fkey(email)
    `)
    .eq('id', docId)
    .single()

  if (error || !data) {
    return null
  }

  return data
}

export default async function ReviewPage({
  params,
}: {
  params: Promise<{ docId: string }>
}) {
  const { docId } = await params
  const document = await getDocument(docId)

  if (!document) {
    notFound()
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Review Document Chunks</h1>
        <p className="mt-2 text-sm text-gray-600">
          Review and approve AI-generated chunks from: {document.title}
        </p>
      </div>

      <ChunkReviewer documentId={docId} />
    </div>
  )
}
