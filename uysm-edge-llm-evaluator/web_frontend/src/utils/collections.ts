/** Build full Qdrant collection names from base name, embed model, and chunk params. */

function safeEmbed(embed: string) {
  return embed.replace(/:/g, '_').replace(/\//g, '_').replace(/\./g, '_')
}

export function collectionNameFull(
  base: string,
  embedModel: string,
  chunkSize: number,
  chunkOverlap: number,
) {
  const safe = safeEmbed(embedModel)
  return `${base}_${safe}_${chunkSize}c_${chunkOverlap}ov`
}

export function smartCollectionNameFull(
  base: string,
  embedModel: string,
  parentSize: number,
  childSize: number,
  childOverlap: number,
) {
  const safe = safeEmbed(embedModel)
  return `${base}_${safe}_${parentSize}p_${childSize}c_${childOverlap}ov`
}

export type RagTypeUi = 'Klasik' | 'Smart' | 'BM25 Klasik' | 'BM25 Smart'

export function ragTypeToFlags(ragType: RagTypeUi) {
  const smartRag = ragType === 'Smart' || ragType === 'BM25 Smart'
  const retrievalMode = ragType.startsWith('BM25') ? 'bm25' : 'vector'
  return { smartRag, retrievalMode }
}
