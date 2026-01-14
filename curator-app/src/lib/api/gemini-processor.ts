import { GoogleGenerativeAI } from '@google/generative-ai'
import { FlowiseChunk } from '@/types'

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || '')


/**
 * Process document directly with Gemini AI
 * Extracts text from document URL and chunks it intelligently
 */
export async function processDocumentWithGemini(
  storageUrl: string,
  docType: string,
  filters: string[]
): Promise<FlowiseChunk[]> {
  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' })

    // First, extract text from the document
    const extractPrompt = `Extract all the text content from this document URL: ${storageUrl}

Please provide the complete text content, maintaining the original structure and formatting as much as possible. Include headers, sections, and all readable text.`

    const extractResult = await model.generateContent(extractPrompt)
    const extractedText = extractResult.response.text()

    if (!extractedText || extractedText.trim().length === 0) {
      throw new Error('Failed to extract text from document')
    }

    // Apply filters to remove unwanted sections
    let filteredText = extractedText
    const filterActions = {
      toc: () => filteredText = removeTableOfContents(filteredText),
      index: () => filteredText = removeIndex(filteredText),
      cover: () => filteredText = removeCoverPages(filteredText),
      boilerplate: () => filteredText = removeBoilerplate(filteredText),
      code: () => filteredText = removeCodeExamples(filteredText),
      appendix: () => filteredText = removeAppendices(filteredText),
    }

    filters.forEach(filter => {
      if (filterActions[filter as keyof typeof filterActions]) {
        filterActions[filter as keyof typeof filterActions]()
      }
    })

    // Chunk the filtered text
    const chunks = await chunkTextWithGemini(filteredText, docType)

    return chunks
  } catch (error) {
    console.error('Error processing document with Gemini:', error)
    throw error
  }
}

/**
 * Intelligent text chunking using Gemini AI
 */
async function chunkTextWithGemini(text: string, docType: string): Promise<FlowiseChunk[]> {
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' })

  const chunkPrompt = `You are an expert document analyzer. Your task is to intelligently chunk the following ${docType} document text into meaningful, self-contained sections.

Guidelines for chunking:
1. Each chunk should be a complete, coherent section (300-800 words)
2. Preserve logical document structure (sections, subsections, topics)
3. Ensure chunks can stand alone with meaningful context
4. Avoid cutting sentences or paragraphs in the middle
5. Focus on semantic coherence over arbitrary length limits

Document text:
${text}

Please return a JSON array of chunks with this exact format:
[
  {
    "index": 0,
    "text": "chunk text here",
    "filtered": false,
    "section_type": "introduction|body|conclusion|etc"
  }
]

Return ONLY the JSON array, no additional text.`

  try {
    const result = await model.generateContent(chunkPrompt)
    const response = result.response.text()

    // Clean the response to extract JSON
    const jsonMatch = response.match(/\[[\s\S]*\]/)
    if (!jsonMatch) {
      throw new Error('No JSON array found in Gemini response')
    }

    const chunks = JSON.parse(jsonMatch[0]) as FlowiseChunk[]

    // Validate and clean chunks
    return chunks.map((chunk, idx) => ({
      index: idx,
      text: chunk.text.trim(),
      filtered: false,
      section_type: chunk.section_type || 'body',
      filter_reason: undefined
    })).filter(chunk => chunk.text.length > 50) // Remove very small chunks

  } catch (error) {
    console.error('Error chunking text with Gemini:', error)
    // Fallback to simple chunking if AI fails
    return fallbackChunking(text)
  }
}

/**
 * Fallback chunking method if AI processing fails
 */
function fallbackChunking(text: string): FlowiseChunk[] {
  const chunkSize = 1000 // characters
  const chunks: FlowiseChunk[] = []
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0)

  let currentChunk = ''
  let chunkIndex = 0

  for (const sentence of sentences) {
    if ((currentChunk + sentence).length > chunkSize && currentChunk.length > 0) {
      chunks.push({
        index: chunkIndex++,
        text: currentChunk.trim(),
        filtered: false,
        section_type: 'body'
      })
      currentChunk = sentence + '. '
    } else {
      currentChunk += sentence + '. '
    }
  }

  if (currentChunk.trim().length > 0) {
    chunks.push({
      index: chunkIndex,
      text: currentChunk.trim(),
      filtered: false,
      section_type: 'body'
    })
  }

  return chunks
}

/**
 * Filter helper functions
 */
function removeTableOfContents(text: string): string {
  // Simple regex-based removal - could be enhanced with AI
  const tocPatterns = [
    /table of contents[\s\S]*?(?=\n\s*\n\s*[A-Z])/i,
    /contents[\s\S]*?(?=\n\s*\n\s*[A-Z])/i,
    /\bTOC\b[\s\S]*?(?=\n\s*\n\s*[A-Z])/i
  ]

  let filtered = text
  tocPatterns.forEach(pattern => {
    filtered = filtered.replace(pattern, '')
  })

  return filtered
}

function removeIndex(text: string): string {
  // Remove index/glossary sections
  const indexPatterns = [
    /index[\s\S]*?(?=\n\s*\n\s*[A-Z])/i,
    /glossary[\s\S]*?(?=\n\s*\n\s*[A-Z])/i
  ]

  let filtered = text
  indexPatterns.forEach(pattern => {
    filtered = filtered.replace(pattern, '')
  })

  return filtered
}

function removeCoverPages(text: string): string {
  // Remove cover page content (usually at the beginning)
  const lines = text.split('\n')
  const contentStart = lines.findIndex(line =>
    line.trim().length > 0 &&
    !line.toLowerCase().includes('cover') &&
    !line.toLowerCase().includes('title') &&
    !/^\s*\d+\s*$/.test(line) // Page numbers
  )

  return lines.slice(contentStart).join('\n')
}

function removeBoilerplate(text: string): string {
  // Remove legal boilerplate, disclaimers, etc.
  const boilerplatePatterns = [
    /this document.*confidential/i,
    /copyright.*all rights reserved/i,
    /disclaimer[\s\S]*?(?=\n\s*\n)/i,
    /legal notice[\s\S]*?(?=\n\s*\n)/i
  ]

  let filtered = text
  boilerplatePatterns.forEach(pattern => {
    filtered = filtered.replace(pattern, '')
  })

  return filtered
}

function removeCodeExamples(text: string): string {
  // Remove code blocks and examples
  return text.replace(/```[\s\S]*?```/g, '')
}

function removeAppendices(text: string): string {
  // Remove appendix sections
  const appendixPattern = /appendix[\s\S]*$/i
  return text.replace(appendixPattern, '')
}

/**
 * Check if direct Gemini processing is available
 */
export function isDirectGeminiConfigured(): boolean {
  return !!(process.env.GOOGLE_API_KEY)
}