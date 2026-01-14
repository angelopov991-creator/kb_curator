import { createClient } from '@/lib/supabase/client';
import { GoogleGenerativeAI } from '@google/generative-ai';

import { OpenAI } from 'openai';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || '');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || '' });

/**
 * Get active AI provider from settings
 */
async function getActiveProvider(): Promise<'gemini' | 'openai'> {
  const supabase = createClient();
  const { data, error } = await supabase
    .from('settings')
    .select('value')
    .eq('key', 'ai_provider')
    .single();

  if (error || !data) {
    return 'gemini'; // Default
  }

  return (data.value as { provider: 'gemini' | 'openai' }).provider || 'gemini';
}

/**
 * Intent Classification
 * Determines which knowledge base(s) to query
 */
async function classifyIntent(query: string): Promise<string[]> {
  const provider = await getActiveProvider();

  if (provider === 'openai') {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{
        role: 'system',
        content: `You are a query classifier for a rural healthcare knowledge base system.
        
Available knowledge bases:
- fhir: FHIR specifications, interoperability, health data exchange
- vbc: Value-based care, quality measures, ACOs, MIPS, population health
- grants: Grant programs, funding opportunities, application guidance
- billing: Medical billing, CPT/ICD coding, revenue cycle, reimbursement
- it_security: Healthcare IT, HIPAA, cybersecurity, EHR systems
- operations: Rural healthcare operations, CAH, RHC, workforce, telemedicine
- compliance: Regulations, CMS requirements, licensing, legal compliance

Classify the query into one or more relevant knowledge bases.
Return ONLY a JSON array of KB names, e.g., ["fhir", "it_security"]`
      }, {
        role: 'user',
        content: query
      }],
      temperature: 0
    });

    const content = response.choices[0].message.content || '[]';
    try {
      const kbs = JSON.parse(content.replace(/```json|```/g, '').trim());
      return Array.isArray(kbs) && kbs.length > 0 ? kbs : ['grants'];
    } catch {
      return ['grants'];
    }
  }

  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
  
  const prompt = `You are a query classifier for a rural healthcare knowledge base system.
      
Available knowledge bases:
- fhir: FHIR specifications, interoperability, health data exchange
- vbc: Value-based care, quality measures, ACOs, MIPS, population health
- grants: Grant programs, funding opportunities, application guidance
- billing: Medical billing, CPT/ICD coding, revenue cycle, reimbursement
- it_security: Healthcare IT, HIPAA, cybersecurity, EHR systems
- operations: Rural healthcare operations, CAH, RHC, workforce, telemedicine
- compliance: Regulations, CMS requirements, licensing, legal compliance

Classify the query into one or more relevant knowledge bases.
Return ONLY a JSON array of KB names, e.g., ["fhir", "it_security"]

Query: ${query}`;

  const result = await model.generateContent(prompt);
  const response = await result.response;
  const content = response.text() || '[]';
  
  try {
    const kbs = JSON.parse(content.replace(/```json|```/g, '').trim());
    // Default to grants if unclear (most common query)
    return Array.isArray(kbs) && kbs.length > 0 ? kbs : ['grants'];
  } catch (e) {
    console.error('Error parsing intent classification:', e);
    return ['grants'];
  }
}

/**
 * Generate embedding for query
 */
async function generateEmbedding(text: string): Promise<number[]> {
  const provider = await getActiveProvider();

  if (provider === 'openai') {
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text
    });
    return response.data[0].embedding;
  }

  const model = genAI.getGenerativeModel({ model: 'text-embedding-004' });
  const result = await model.embedContent(text);
  return result.embedding.values;
}

/**
 * Query single knowledge base
 */
async function queryKnowledgeBase(
  queryEmbedding: number[],
  docType: string,
  limit: number = 5
) {
  const provider = await getActiveProvider();
  const supabase = createClient();
  const { data, error } = await supabase.rpc('match_documents', {
    query_embedding: queryEmbedding,
    match_threshold: 0.7,
    match_count: limit,
    filter_doc_type: docType,
    provider: provider
  });

  if (error) {
    console.error(`Error querying ${docType} KB:`, error);
    return [];
  }

  return data || [];
}

/**
 * Main RAG Query Function
 */
export async function ragQuery(
  userQuery: string,
  options?: {
    maxChunks?: number;
    includeMetadata?: boolean;
  }
) {
  const maxChunks = options?.maxChunks || 10;

  // Step 1: Classify intent
  const relevantKBs = await classifyIntent(userQuery);
  
  // Step 2: Generate embedding
  const queryEmbedding = await generateEmbedding(userQuery);
  
  // Step 3: Query relevant KBs in parallel
  const chunksPerKB = Math.ceil(maxChunks / relevantKBs.length);
  
  const results = await Promise.all(
    relevantKBs.map(kb => queryKnowledgeBase(queryEmbedding, kb, chunksPerKB))
  );
  
  // Step 4: Flatten and rank results
  const allChunks = results.flat();
  const rankedChunks = allChunks
    .sort((a: { similarity: number }, b: { similarity: number }) => b.similarity - a.similarity)
    .slice(0, maxChunks);
  
  return {
    chunks: rankedChunks,
    relevantKBs,
    totalResults: allChunks.length
  };
}

/**
 * Agent Implementations
 *
 * 1. Gap Analysis Agent
 * Purpose: Identify gaps between current state and desired state
 *
 * Flow:
 * User Input: Current state + Desired outcome
 *     ↓
 * Extract key topics
 *     ↓
 * Query relevant KBs (VBC, Operations, IT, Compliance)
 *     ↓
 * Identify missing capabilities
 *     ↓
 * Generate gap analysis report
 */