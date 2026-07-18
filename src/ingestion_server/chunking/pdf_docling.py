"""PDF -> chunks via Docling, with full provenance (page_no + bbox).

Ported from noted's graph/app/scanners/pdf_scanner.py, which was tuned
against this very CV over many iterations — the heading-hierarchy rebuild,
the per-bullet split for the certifications list, the per-item (not
per-page union) regions, and the super-parent chunks all exist because of
concrete failures on this document. Rewriting that from scratch would have
regressed retrieval, so it is carried over deliberately.

Adapted to stand alone: noted's config/profile/model imports are inlined
below, and the picture/table captioning pass is stubbed out (it was gated
OFF for this corpus, so it never contributed a chunk). Nothing here talks
to noted at runtime.

Docling's converter and chunker are expensive to build (the bge-m3
tokenizer dominates), so both are constructed lazily and reused for the
process lifetime.
"""

from __future__ import annotations

import logging
import os
import re

from dataclasses import dataclass, replace
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Inlined from noted's app/config.py ────────────────────────────────
# Docling's PdfPipelineOptions.do_ocr defaults True, which eagerly loads
# RapidOCR weights we don't ship. The CV is a text PDF we render
# ourselves, so OCR stays off; scan_pdf still self-heals to an
# OCR-enabled converter if an extraction comes back near-empty.
DOCLING_OCR = os.environ.get('DOCLING_OCR', 'false').lower() in ('1', 'true', 'yes')
DOCLING_TABLE_MODE = os.environ.get('DOCLING_TABLE_MODE', 'accurate')
DOCLING_MAX_PAGES = int(os.environ.get('DOCLING_MAX_PAGES', '2000'))
# The captioning pass is not ported (see the stubs at the bottom).
ENABLE_DOC_DESCRIPTIONS = False

# Model roots. Defaults match the bind mounts the ingest container gets:
# CV's own docling copy (cv/data/models/docling) and bge-m3 for the
# chunker's tokenizer. Both guarded by isdir() at the use site, which
# falls back to downloading from HuggingFace.
DOCLING_ARTIFACTS_ROOT = os.environ.get('DOCLING_ARTIFACTS_ROOT', '/models/docling/models')
BGE_M3_LOCAL = os.environ.get('BGE_M3_LOCAL', '/models/bge-m3')


# ── Inlined from noted's app/chunking_profiles.py ─────────────────────
# Only the profile this pipeline uses is carried over. noted's reingest
# asked for `code-docs` (target 300) — but noted's own catalog labels
# `list-dense` as "List-dense (CVs, bullet docs)", which is plainly the
# right profile for a bullet-packed CV. Docling-style chunkers honour
# only max_tokens, and noted fed them `target_tokens`, so this collapses
# to a single knob: HybridChunker(max_tokens=...).
CHUNK_TARGET_TOKENS = int(os.environ.get('CHUNK_TARGET_TOKENS', '200'))


def resolve_chunking_profile(profile_id: str | None = None) -> dict:
    return {'id': 'list-dense', 'target_tokens': CHUNK_TARGET_TOKENS}


# ── Inlined from noted's app/scanners/md_scanner.py ───────────────────
# Not frozen: the bullet-split and super-parent passes re-sequence
# chunk_index in place after they add chunks.
@dataclass
class MdChunk:
    doc_path: str            # path relative to the corpus root
    chunk_index: int         # 0-based within the doc
    section_path: str        # e.g. "CERTIFICATIONS > AI, Machine Learning, Data"
    text: str
    token_count: int
    page_no: int | None = None       # 1-indexed
    bbox: list[float] | None = None  # [x0,y0,x1,y1], PDF space (bottom-left origin)
    section_level: int | None = None
    # A chunk can span pages (HybridChunker merges adjacent doc_items across
    # page breaks), so `regions` carries one bbox per item per page and
    # page_no/bbox mirror regions[0].
    regions: list[dict] | None = None
    kind: str = 'text'
    caption_for: str | None = None

# Drop chunks whose underlying doc_items are mostly TOC / index entries.
# Docling's layout model already classifies these as DocItemLabel.DOCUMENT_INDEX,
# so we just filter on the label — no string heuristics, no language coupling.
# Default ratio = 1.0 means "only drop chunks that are 100 % DOCUMENT_INDEX
# items" (strict, near-zero false positives). Lower to 0.5 for majority vote
# if mixed-content TOC merges become a problem in practice. Set to 0.0
# (or any negative value) to disable the filter entirely.
TOC_FILTER_RATIO: float = float(os.environ.get('TOC_FILTER_RATIO', '1.0'))


_converter = None
# Chunker is now keyed by max_tokens so different chunking profiles each
# get their own HybridChunker instance, cached for the lifetime of the
# process. The chunker constructor itself is cheap; the expensive bit is
# tokenizer init (bge-m3), which happens once per max_tokens value. With
# 4 profiles in the catalog, the cache holds at most 4 entries.
_chunkers: dict[int, object] = {}


def _ensure_loaded() -> None:
    """Lazy-init Docling converter. Idempotent. Chunkers are built on
    demand by _get_chunker(max_tokens)."""
    global _converter
    if _converter is not None:
        return

    # Imported lazily so module import is free even when no PDFs are
    # ingested in a given process (e.g. during a markdown-only rebuild).
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # PdfPipelineOptions.do_ocr defaults to True, which causes Docling to
    # eagerly load RapidOCR (~40 MB of weights into the container's
    # site-packages, NOT our cache mount). Honor DOCLING_OCR explicitly.
    #
    # artifacts_path: point Docling at the bind-mounted on-disk model
    # cache. Without this, Docling falls back to ~/.cache/huggingface/hub
    # which is NOT bind-mounted, so every container rebuild re-downloads
    # ~3 GB of layout/tableformer/figure-classifier weights from HF. The
    # on-disk layout under /data/models/docling/models/ matches Docling's
    # `artifacts_path` format (`<repo_id_with_dashes>/`), populated once
    # on the host. Guarded with isdir() so missing-mount falls back to
    # the default behavior instead of breaking.
    artifacts_root = DOCLING_ARTIFACTS_ROOT
    pdf_opts = PdfPipelineOptions(
        do_ocr=DOCLING_OCR,
        artifacts_path=artifacts_root if os.path.isdir(artifacts_root) else None,
        # Generate the actual cropped image bytes for every PictureItem so
        # the captioner can send them to the vision LLM. Off by default in
        # Docling because most consumers don't need the bytes; we do when
        # ENABLE_DOC_DESCRIPTIONS is on.
        generate_picture_images=ENABLE_DOC_DESCRIPTIONS,
        # Image scale: the default (1.0) yields ~72 DPI. The vision encoder
        # benefits from a slightly higher resolution for charts / diagrams
        # without ballooning the payload.
        images_scale=2.0 if ENABLE_DOC_DESCRIPTIONS else 1.0,
    )

    _converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.HTML,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        },
    )
    logger.info(
        'Docling loaded: table_mode=%s ocr=%s max_pages=%s',
        DOCLING_TABLE_MODE, DOCLING_OCR, DOCLING_MAX_PAGES,
    )


_ocr_converter = None
# A scanned/image PDF extracts to near-empty text when do_ocr is off. Below
# this many characters we treat the extraction as failed and retry with OCR.
_OCR_FALLBACK_MIN_CHARS = int(os.environ.get('DOCLING_OCR_FALLBACK_MIN_CHARS', '500'))


def _get_ocr_converter():
    """Lazy OCR-enabled converter, built only when a near-empty extraction
    triggers the scanned-PDF fallback (so normal text PDFs never load OCR).
    Same options as the main converter but do_ocr=True. RapidOCR weights load
    on first use."""
    global _ocr_converter
    if _ocr_converter is not None:
        return _ocr_converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    # NB: do NOT pin artifacts_path here — the on-disk cache holds the layout
    # models but NOT the RapidOCR weights, so pinning it makes OCR fail with a
    # missing-model error. Leaving it unset lets Docling resolve RapidOCR (and
    # the layout models, already in the HF cache) from their default locations —
    # the config proven to work in the standalone OCR test.
    pdf_opts = PdfPipelineOptions(
        do_ocr=True,
        generate_picture_images=ENABLE_DOC_DESCRIPTIONS,
        images_scale=2.0 if ENABLE_DOC_DESCRIPTIONS else 1.0,
    )
    _ocr_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX,
                         InputFormat.PPTX, InputFormat.HTML],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)},
    )
    logger.info('Docling OCR-fallback converter loaded (do_ocr=True)')
    return _ocr_converter


def _get_chunker(max_tokens: int):
    """Return a HybridChunker for the given max_tokens, cached for the
    lifetime of the process. Tokenizer init (bge-m3) is the expensive
    part; for the small set of distinct profile max_tokens values in
    practice (4 today) we accept paying it once per value."""
    chunker = _chunkers.get(max_tokens)
    if chunker is not None:
        return chunker
    from docling.chunking import HybridChunker
    # Tokenizer: prefer the on-disk bge-m3 (under the bind-mounted models
    # tree); fall back to the HF id only if the local copy is missing.
    # Without this, HybridChunker calls HuggingFace on every container
    # restart for the tokenizer config files even though the full model
    # is on disk.
    bge_m3_local = BGE_M3_LOCAL
    chunker = HybridChunker(
        tokenizer=bge_m3_local if os.path.isdir(bge_m3_local) else 'BAAI/bge-m3',
        max_tokens=max_tokens,
        merge_peers=True,
    )
    _chunkers[max_tokens] = chunker
    logger.info('HybridChunker built: tokenizer=bge-m3 max_tokens=%d', max_tokens)
    return chunker


def scan_pdf(
    abs_path: str,
    repo_root: str | None = None,
    progress_writer: Callable[[dict], None] | None = None,
    *,
    chunking_profile_id: str | None = None,
) -> list[MdChunk]:
    """Convert one PDF (or DOCX / PPTX / HTML) into MdChunk instances.

    Returned chunks carry the same fields as md_scanner output plus the
    optional provenance trio (page_no, bbox, section_level). Caller
    appends the same Entity + chunked_into edges as for markdown chunks.

    `repo_root` is the Domain's sources/ directory so doc_path comes out
    Domain-relative (e.g. `foo.pdf`). Required in production paths;
    callers compute it via `corpus.sources_dir(domain_id)`. When omitted
    (tests), doc_path falls back to the file's basename.

    `progress_writer` is an optional callable that merges its dict argument
    into the caller's progress dict. Used during the picture/table
    captioning pass to surface live `sub_phase` data + persistent
    `pictures_total` / `tables_total` etc. counters in the KB Monitor.
    Wrapped in try/except so a misbehaving consumer never breaks scanning.

    Raises ValueError if the document exceeds DOCLING_MAX_PAGES (a safety
    cap against runaway uploads).
    """
    _ensure_loaded()
    if repo_root:
        rel_path = os.path.relpath(abs_path, repo_root)
    else:
        rel_path = os.path.basename(abs_path)

    # Resolve chunking profile → max_tokens for the HybridChunker. None
    # falls back to the catalog's default profile (whose target_tokens
    # historically matched EXTRACT_CHUNK_TARGET_TOKENS=600). We feed
    # Docling the profile's `target_tokens` (Docling's `max_tokens` is
    # really an upper bound it tries to stay near, so target ≈ Docling
    # max).
    profile = resolve_chunking_profile(chunking_profile_id)
    docling_max_tokens = profile['target_tokens']
    _chunker = _get_chunker(docling_max_tokens)

    result = _converter.convert(abs_path)
    doc = result.document  # DoclingDocument

    # Self-healing OCR fallback: scanned/image PDFs extract to near-empty text
    # when do_ocr is off. Retry THIS doc with OCR (targeted — text PDFs never
    # pay the OCR cost). Skipped when OCR is already global.
    if not DOCLING_OCR:
        try:
            _txt = (doc.export_to_markdown() or '').strip()
        except Exception:
            _txt = ''
        if len(_txt) < _OCR_FALLBACK_MIN_CHARS:
            logger.info('scan_pdf: near-empty extraction (%d chars) for %s '
                        '-> retrying with OCR', len(_txt), rel_path)
            result = _get_ocr_converter().convert(abs_path)
            doc = result.document

    page_count = len(getattr(doc, 'pages', []) or [])
    if DOCLING_MAX_PAGES and page_count > DOCLING_MAX_PAGES:
        raise ValueError(
            f'page count {page_count} exceeds DOCLING_MAX_PAGES={DOCLING_MAX_PAGES} for {rel_path}'
        )

    # Lazy-import the label enum so module import stays Docling-free.
    from docling_core.types.doc import DocItemLabel

    # ── Picture + table captioning pass (Path A) ──────────────────────
    # When ENABLE_DOC_DESCRIPTIONS is on, walk every PictureItem and
    # TableItem in the converted doc, generate an LLM caption for each,
    # and stash the result in a manifest keyed by the item's `self_ref`.
    # The chunk loop below consults this manifest to inject captions
    # into the appropriate chunks (or emit standalone caption chunks
    # for orphan pictures whose chunks would otherwise be empty).
    caption_manifest: dict[str, dict[str, Any]] = _build_caption_manifest(
        doc, rel_path, progress_writer,
    )

    # Track which manifest entries got consumed by the chunk loop, so we
    # know which orphan items still need standalone chunks at the end.
    consumed_self_refs: set[str] = set()

    # Docling's PDF parser frequently flattens the heading hierarchy:
    # every visual header lands at `level=1`, and HybridChunker's
    # `raw.meta.headings` returns only the chunk's immediate parent.
    # That loses real ancestry — a bullet under "## Certifications →
    # ### AI, Machine Learning, Data" ends up with section_path
    # "AI, Machine Learning, Data" alone, with no trace of the
    # "Certifications" parent. Diana then can't match a "certifications"
    # query against those chunks because the word never appears in any
    # of them. _build_heading_hierarchy walks the doc once to recover
    # parent/child links from positional cues (a heading immediately
    # followed by another heading is a container of the next one), and
    # the section_path computation below uses that map.
    heading_ancestors = _build_heading_hierarchy(doc)

    chunks: list[MdChunk] = []
    n_toc_dropped = 0
    for i, raw in enumerate(_chunker.chunk(doc)):
        text = raw.text or ''
        if not text.strip():
            continue

        # TOC / document-index filter. Drops chunks whose constituent
        # doc_items are mostly DOCUMENT_INDEX (Docling's TOC label). At
        # the default TOC_FILTER_RATIO=1.0, only fully-TOC chunks are
        # skipped; raise/lower via env var. Indexes pollute retrieval
        # (no semantic content) and graph extraction (one entity per
        # heading) without adding signal.
        items = raw.meta.doc_items or []
        if items and TOC_FILTER_RATIO > 0:
            n_toc = sum(1 for it in items if getattr(it, 'label', None) == DocItemLabel.DOCUMENT_INDEX)
            if n_toc / len(items) >= TOC_FILTER_RATIO:
                n_toc_dropped += 1
                continue

        first_item = raw.meta.doc_items[0] if raw.meta.doc_items else None
        prov = first_item.prov[0] if first_item and first_item.prov else None
        raw_headings = list(raw.meta.headings or [])
        # Enrich the heading chain with reconstructed ancestors so the
        # section_path carries the full parent → child path Docling lost.
        direct_heading = raw_headings[-1] if raw_headings else None
        if direct_heading and direct_heading in heading_ancestors:
            full_chain = heading_ancestors[direct_heading] + raw_headings
        else:
            full_chain = raw_headings
        section_path = ' > '.join(full_chain) or '(root)'

        # HybridChunker(merge_peers=True) merges adjacent doc_items into one
        # chunk. Three consequences for bbox computation:
        #   1. Marginalia labels and cross-reference glyphs are extracted by
        #      Docling as their own (tiny) doc_items and get merged with the
        #      body paragraph. Taking only the first prov's bbox highlighted
        #      a 50x5pt margin label instead of the body.
        #   2. A merged chunk can span page breaks. The body of one
        #      paragraph may end on page N+1 while its earlier doc_items
        #      sit on page N.
        #   3. A chunk can mix prose with a PictureItem (figure embedded
        #      in a section). Unioning the picture's bbox into the prose
        #      union inflates the highlight to span "top of paragraph →
        #      bottom of figure", looking broken.
        # Strategy: per page, emit ONE region for the union of TEXT items,
        # PLUS one region per PICTURE item (un-unioned with text or with
        # other pictures). Tables stay in the text union — they contribute
        # markdown content to the chunk text and the table-region is the
        # right thing to highlight. The deep-jump highlight paints every
        # region as a separate rectangle, so a section + figure becomes
        # two rectangles on the same page rather than one giant box.
        # `page_no`/`bbox` mirror regions[0] for callers that still consume
        # the single-region fields.
        regions: list[dict] | None = None
        bbox = None
        page_no = None
        text_bboxes_per_page: dict[int, list[tuple[float, float, float, float]]] = {}
        picture_bboxes_per_page: dict[int, list[tuple[float, float, float, float]]] = {}
        first_page_seen: int | None = None
        for item in (raw.meta.doc_items or []):
            is_picture = getattr(item, 'label', None) == DocItemLabel.PICTURE
            for p in (getattr(item, 'prov', None) or []):
                p_page = getattr(p, 'page_no', None)
                bb = getattr(p, 'bbox', None)
                if p_page is None or bb is None:
                    continue
                try:
                    tup = tuple(float(v) for v in bb.as_tuple())
                except Exception:
                    continue
                page_int = int(p_page)
                if is_picture:
                    picture_bboxes_per_page.setdefault(page_int, []).append(tup)
                else:
                    text_bboxes_per_page.setdefault(page_int, []).append(tup)
                if first_page_seen is None:
                    first_page_seen = page_int
        if text_bboxes_per_page or picture_bboxes_per_page:
            # Region order: first page seen first (matches reading order),
            # then any other pages sorted ascending. Within each page emit
            # ONE region per text item (in encounter order) plus one region
            # per picture item.
            #
            # Earlier this unioned all text items per page into a single
            # bbox. That works when the items are contiguous, but docling
            # sometimes lumps geometrically-distant items into the same
            # chunk (e.g. a stray right-column header at the top of a page
            # ending up bagged with the actual content lower in the column).
            # The union then spans both items and the white space between,
            # producing a "highlight that doesn't enclose any concrete
            # text" effect. Per-item regions outline each actual line/
            # paragraph instead, so the user sees the precise text the
            # chunk references — even when an item is misgrouped, it shows
            # up as its own small box, not as part of a giant rectangle.
            all_pages = set(text_bboxes_per_page) | set(picture_bboxes_per_page)
            other_pages = sorted(p for p in all_pages if p != first_page_seen)
            ordered_pages = (
                [first_page_seen] + other_pages if first_page_seen is not None
                else sorted(all_pages)
            )
            regions = []
            for p_page in ordered_pages:
                for tb in (text_bboxes_per_page.get(p_page) or []):
                    regions.append({
                        'page_no': p_page,
                        'bbox': [tb[0], tb[1], tb[2], tb[3]],
                    })
                for pb in (picture_bboxes_per_page.get(p_page) or []):
                    regions.append({
                        'page_no': p_page,
                        'bbox': [pb[0], pb[1], pb[2], pb[3]],
                    })
            if regions:
                # `page_no` / `bbox` denormalised fields on the chunk now
                # mirror the FIRST region for callers that still consume
                # the legacy single-rect fields. The full geometry lives
                # in regions[].
                page_no = regions[0]['page_no']
                bbox = regions[0]['bbox']

        section_level = getattr(raw.meta, 'headings_level', None)

        # Inject captions for any PictureItem/TableItem this chunk
        # contains. Caption text is prepended so it acts as framing for
        # the (sparse) cell content or surrounding prose. `kind` flips
        # to 'picture_caption' / 'table_caption' only when the chunk is
        # exclusively that one item — i.e. the caption IS the chunk's
        # primary content. Mixed chunks (table + paragraph, picture +
        # text) stay 'text' because the prose framing remains the
        # dominant content; the caption is supplementary.
        chunk_kind, caption_for, caption_text = _resolve_chunk_caption(
            raw.meta.doc_items or [], caption_manifest,
        )
        if caption_text:
            text = caption_text + '\n\n' + text
            for it in (raw.meta.doc_items or []):
                ref = getattr(it, 'self_ref', None)
                if ref and ref in caption_manifest:
                    consumed_self_refs.add(ref)

        chunks.append(MdChunk(
            doc_path=rel_path,
            chunk_index=i,
            section_path=section_path,
            text=text,
            token_count=_chunker.tokenizer.count_tokens(text),
            page_no=int(page_no) if page_no is not None else None,
            bbox=bbox,
            regions=regions,
            section_level=int(section_level) if section_level is not None else None,
            kind=chunk_kind,
            caption_for=caption_for,
        ))

    # ── Orphan caption chunks ──────────────────────────────────────────
    # Pictures with no surrounding text get dropped by the chunker (text
    # was empty). Emit standalone caption chunks for them so the picture
    # is still indexable, citable, and bbox-highlightable. Same applies
    # to any TableItem whose chunk got filtered out — rare but possible.
    next_idx = len(chunks)
    for self_ref, entry in caption_manifest.items():
        if self_ref in consumed_self_refs:
            continue
        prov = entry.get('prov')
        page_no = prov.get('page_no') if prov else None
        bbox = prov.get('bbox') if prov else None
        regions = [{'page_no': page_no, 'bbox': bbox}] if (page_no and bbox) else None
        caption = entry.get('caption') or ''
        if not caption.strip():
            continue
        chunks.append(MdChunk(
            doc_path=rel_path,
            chunk_index=next_idx,
            section_path=entry.get('section_path') or '(root)',
            text=caption,
            token_count=_chunker.tokenizer.count_tokens(caption),
            page_no=int(page_no) if page_no is not None else None,
            bbox=bbox,
            regions=regions,
            section_level=None,
            kind=entry.get('kind') or 'text',
            caption_for=self_ref,
        ))
        next_idx += 1

    # ── Per-bullet split for list-only chunks ─────────────────────────
    # Docling's HybridChunker bags a whole bulleted list into ONE chunk
    # (each bullet too short on its own to pass min_tokens). The bbox/
    # regions per chunk then point to ALL bullets in the list — clicking
    # "Azure AI Fundamentals AI-901" highlights the entire 7-bullet cert
    # list, and Diana cites a single tag (e.g. "1") for all 7 items.
    #
    # Split each bullet-only chunk into N per-bullet chunks: each carries
    # ONE bullet's text + ONE region. We only split when the line count
    # matches the region count (1-to-1 docling mapping); otherwise we
    # keep the chunk intact rather than guess which region goes with
    # which bullet.
    chunks = _split_bullet_chunks(chunks, _chunker.tokenizer, doc=doc,
                                  heading_ancestors=heading_ancestors,
                                  rel_path=rel_path)

    # ── Super-parent chunks across sibling subsections ────────────────
    # The per-subsection parent + per-bullet sub-chunks are precise but
    # short. Verbose queries like "antónio certifications and
    # qualifications" rerank short list-shaped chunks poorly even when
    # the right hierarchy is in their text. The fix is to also emit one
    # WIDER chunk per top-level container heading (e.g. "CERTIFICATIONS"
    # spans AI/ML/Data + Architecture + PM/Agile, "Vision-Box · Portugal"
    # spans CDO + CTO + R&D Director). Its text concatenates every
    # sibling subsection's content; its regions[] is the union of every
    # sibling's regions — so clicking the citation still highlights real
    # rectangles, just many of them at once. The reranker then chooses
    # the right granularity per query: broad questions favour the
    # super-parent (~all bullets in one place), specific questions still
    # win with the per-bullet sub-chunk.
    chunks = _add_super_parent_chunks(chunks, _chunker.tokenizer)

    # Guarantee no chunk exceeds the embedder. Docling keeps an atomic element
    # (a large table/list) as one chunk, and _add_super_parent_chunks can
    # concatenate siblings — either can blow past bge-m3's context. Since
    # noted-rag embeds a doc's chunks in one ATOMIC call, a single oversized
    # chunk fails the WHOLE document's embed (it lands in the graph but never
    # the vector corpus). Hard-split the oversized tail so that can't happen.
    chunks = _split_oversized_chunks(chunks)

    logger.info(
        'pdf scan: %s -> %d chunks across %d pages (toc_dropped=%d, ratio=%.2f, captions=%d)',
        rel_path, len(chunks), page_count, n_toc_dropped, TOC_FILTER_RATIO,
        sum(1 for c in chunks if c.kind in ('picture_caption', 'table_caption')),
    )
    return chunks


# Conservative CHARACTER cap (tokens are always fewer than chars here, so this
# stays well under bge-m3's 4096-token slot even for token-dense tables).
# Only the ~0.4% oversized tail is affected; normal chunks (p99 ~3.9k chars)
# pass through untouched.
_MAX_CHUNK_CHARS = int(os.environ.get('MAX_CHUNK_CHARS', '6000'))
_CHUNK_SPLIT_OVERLAP = int(os.environ.get('CHUNK_SPLIT_OVERLAP', '400'))


def _split_oversized_chunks(chunks: list[MdChunk]) -> list[MdChunk]:
    """Window any chunk whose text exceeds `_MAX_CHUNK_CHARS` into overlapping
    sub-chunks so no single chunk can exceed the embedder. Sub-chunks inherit
    the parent's provenance (page/bbox/regions approximate to the parent's span
    — a split table still highlights its region). `chunk_index` is re-sequenced
    so per-doc chunk ids stay unique."""
    cap, overlap = _MAX_CHUNK_CHARS, _CHUNK_SPLIT_OVERLAP
    step = max(1, cap - overlap)
    out: list[MdChunk] = []
    for c in chunks:
        t = c.text or ''
        if len(t) <= cap:
            out.append(c)
            continue
        for start in range(0, len(t), step):
            part = t[start:start + cap]
            if part.strip():
                out.append(replace(c, text=part, token_count=len(part) // 4))
            if start + cap >= len(t):
                break
    return [replace(c, chunk_index=i) for i, c in enumerate(out)]


# Lines starting with a bullet glyph (hyphen, asterisk, bullet, en-dash)
# plus a space, optionally indented. Matches the bullet shapes Docling
# emits when normalising lists to markdown.
_BULLET_LINE_RE = re.compile(r'^\s*(?:[-*•–]|[•‣◦])\s+\S')


def _add_super_parent_chunks(chunks: list[MdChunk], tokenizer) -> list[MdChunk]:
    """Emit one extra wide chunk per top-level container heading.

    For each unique outermost ancestor (the part of section_path before
    the first ' > '), gather every direct subsection-parent chunk that
    sits under it and produce one combined chunk whose text is all
    sibling content stitched together and whose regions[] is the union
    of every sibling's bbox list. This is the level of detail the old
    HTML/markdown pipeline produced naturally and that PDF ingestion
    fragmented — it gives the reranker a high-signal, multi-sentence
    target for category-wide questions ("what are his certifications").

    Sub-chunks emitted by `_split_bullet_chunks` are excluded from the
    fanout — their text already starts with `<section_path>\\n` so they
    are easy to identify, and including them would double-count bullets.

    Groups of size < 2 are skipped: a container with only one child
    already has a subsection-parent chunk that covers the same span.

    chunk_index is renumbered globally so downstream chunk_id stays
    unique. Token count uses the chunker's tokenizer for honest sizing
    even when text concatenation pushes a chunk past the usual target.
    """
    from collections import defaultdict

    groups: dict[str, list[MdChunk]] = defaultdict(list)
    for c in chunks:
        if c.kind != 'text' or not c.section_path or ' > ' not in c.section_path:
            continue
        # Sub-chunks created by _split_bullet_chunks start with the
        # section_path heading line; skip them so we only union the
        # subsection-PARENT chunks (which contain the whole bullet list
        # in one block).
        if c.text.startswith(c.section_path + '\n'):
            continue
        top = c.section_path.split(' > ', 1)[0]
        groups[top].append(c)

    extras: list[MdChunk] = []
    for top, siblings in groups.items():
        if len(siblings) < 2:
            continue
        combined_text = top + '\n\n' + '\n\n'.join(s.text for s in siblings)
        combined_regions: list[dict] = []
        for s in siblings:
            for r in (s.regions or []):
                combined_regions.append(r)
        if not combined_regions:
            continue
        first_region = combined_regions[0]
        extras.append(MdChunk(
            doc_path=siblings[0].doc_path,
            chunk_index=0,
            section_path=top,
            text=combined_text,
            token_count=tokenizer.count_tokens(combined_text),
            page_no=first_region.get('page_no'),
            bbox=first_region.get('bbox'),
            regions=combined_regions,
            section_level=siblings[0].section_level,
            kind='text',
            caption_for=None,
        ))

    out = chunks + extras
    for i, c in enumerate(out):
        c.chunk_index = i
    return out


_SECTION_NUM_RE = re.compile(r'^\s*(\d+(?:\.\d+)*)\.?\s+\S')


def _section_depth(text: str) -> int | None:
    """Depth implied by a leading section number: '3.' -> 1, '2.1' -> 2.

    None when the heading isn't numbered. Generic: numbered sections are a
    convention of reports, standards and manuals, not of any one corpus.
    """
    m = _SECTION_NUM_RE.match(text or '')
    return len(m.group(1).split('.')) if m else None


def _build_heading_hierarchy(doc) -> dict[str, list[str]]:
    """Reconstruct heading ancestry from positional cues.

    Docling's PDF backend assigns the same `level` to every section_header
    in CV-style documents, so the chunker can't tell that
    "AI, Machine Learning, Data" is a child of "Certifications". This
    walks the doc body once and treats two patterns as ancestry signals:

    1. **Containers.** A heading whose IMMEDIATELY-following body item
       is also a heading has no content of its own and is therefore a
       parent of that next heading (and likely of further siblings
       until the section ends).
    2. **All-uppercase section markers.** ALL-CAPS headings (e.g.
       "EDUCATION", "CERTIFICATIONS") are treated as top-level section
       boundaries — they reset the container stack so that the next
       set of containers belongs to the new section, not the previous
       one.

    A container's "range" extends from its own position to the next
    container OR the next all-uppercase heading (whichever comes
    first). Every heading inside that range is treated as a descendant
    of the container.

    Returns: { heading_text → [ancestor1_text, ancestor2_text, ...] }
    ordered outermost-first. Self is NEVER included.

    Note: deep nesting where a container's child is itself a container
    of further siblings (e.g. EDUCATION > ISCTE > {Postgraduate,
    Coimbra}) is approximate — without true level info we'll sometimes
    attach a sibling as a deeper child. That's still a strict
    improvement over docling's flat single-heading output for the
    queries that actually matter (cert + role lookups).
    """
    items = []
    for item, _depth in doc.iterate_items():
        label = str(getattr(item, 'label', ''))
        is_h = 'header' in label.lower() or 'title' in label.lower()
        text = (getattr(item, 'text', '') or '').strip()
        # Geometry (page, y_top), for the two-column fallback below. y is
        # bottom-left origin, so a LARGER y_top is HIGHER on the page.
        page = y_top = None
        for p in (getattr(item, 'prov', None) or []):
            bb = getattr(p, 'bbox', None)
            pg = getattr(p, 'page_no', None)
            if bb is not None and pg is not None:
                try:
                    page, y_top = int(pg), float(bb.as_tuple()[3])
                except Exception:
                    pass
                break
        items.append((is_h, text, page, y_top))

    # Containers: heading positions whose next item is also a heading.
    container_positions = set()
    for i, (is_h, _t, _pg, _yt) in enumerate(items):
        if is_h and i + 1 < len(items) and items[i + 1][0]:
            container_positions.add(i)

    # Pre-compute each container's range end: the next heading that is either
    # another container, an all-uppercase section marker, or a SIBLING by
    # section numbering.
    def _range_end(i: int) -> int:
        own = _section_depth(items[i][1])
        for j in range(i + 1, len(items)):
            is_h, text = items[j][0], items[j][1]
            if not is_h:
                continue
            if j in container_positions or text.isupper():
                return j
            # A heading numbered at the same depth (or shallower) is a sibling,
            # not a child, so the container's range stops here. Without this, a
            # document with no ALL-CAPS markers has nothing to close a range:
            # measured on a 10-page report, "2. Environment Analysis" swallowed
            # "3. Experimental Setup" and all of its subsections. Verified to
            # change nothing on the CV or on three bulas (they use ALL-CAPS, so
            # the caps rule closes ranges first).
            other = _section_depth(text)
            if own is not None and other is not None and other <= own:
                return j
        return len(items)

    container_range = {i: _range_end(i) for i in container_positions}

    # For every heading, gather ancestor containers whose range covers
    # it, ordered by position (outermost-first).
    out: dict[str, list[str]] = {}
    for i, (is_h, text, _pg, _yt) in enumerate(items):
        if not is_h or not text:
            continue
        ancestors: list[str] = []
        for c_pos in sorted(container_positions):
            if c_pos >= i:
                break
            if c_pos < i < container_range[c_pos]:
                ancestors.append(items[c_pos][1])
        # Drop accidental self-reference (a container heading would
        # otherwise see itself as its own ancestor through any outer
        # container that wraps it — that's fine, but exclude its own
        # text just in case of a repeated label).
        ancestors = [a for a in ancestors if a != text]
        out[text] = ancestors

    # ── Two-column geometric fallback ─────────────────────────────────
    # Reading order breaks on multi-column pages: Docling emits the LEFT column
    # top-to-bottom, hits the next full-width ALL-CAPS section, and only THEN
    # emits the RIGHT column's subsection — which therefore appears AFTER that
    # section in reading order and loses its parent via the range logic above.
    # Measured on the CV: "Architecture, Cloud, Engineering" (right column, under
    # CERTIFICATIONS) came out with NO ancestor because "SELECTED ACHIEVEMENTS"
    # (a later, lower section) was emitted between the columns.
    #
    # Fix by GEOMETRY, which reading order can't see: a subsection with no parent
    # is assigned to the nearest ALL-CAPS section that sits physically ABOVE it
    # (earlier page, or same page with a higher y_top). Only fills EMPTY ancestry,
    # so it can never disturb a heading the range logic already resolved.
    caps = [(t, pg, yt) for (h, t, pg, yt) in items
            if h and t and t.isupper() and any(c.isalpha() for c in t)
            and pg is not None and yt is not None]
    for (is_h, text, pg, yt) in items:
        if not is_h or not text or out.get(text) or pg is None or yt is None:
            continue
        if text.isupper():          # a top-level section has no parent
            continue
        best = None                 # (distance, section_text)
        for (st, spg, syt) in caps:
            if st == text:
                continue
            above = spg < pg or (spg == pg and syt >= yt)
            if not above:
                continue
            dist = (pg - spg) * 100000 + (syt - yt if spg == pg else 0)
            if best is None or dist < best[0]:
                best = (dist, st)
        if best:
            out[text] = [best[1]]
    return out


def _split_bullet_chunks(chunks: list[MdChunk], tokenizer, doc=None,
                         heading_ancestors: dict[str, list[str]] | None = None,
                         rel_path: str = '') -> list[MdChunk]:
    """Keep every parent chunk (whole list — rich retrieval signal) and ADD one
    sub-chunk per bullet (single-region, precise highlight target).

    Both granularities coexist and the model picks at runtime: cite the parent to
    highlight a whole category, cite the sub-chunk to highlight one line.

    **Bullets come from Docling's document model, not from re-splitting the
    chunker's text.** Measured on the CV: the tree yields 45 LIST_ITEMs, all 45
    with their own bbox, and they are a strict SUPERSET of what line-matching
    found (34 shared, 11 extra, 0 missed). The old approach only split when line
    count equalled region count and bailed otherwise — "rather than guess which
    bullet goes with which region" — so 11 bullets stayed lumped and
    uncitable. The model already knows; there is nothing to guess.

    What Docling does NOT give (verified four ways) is ancestry: every list item
    sits at level 2, its parents are an unnamed `list` group and `unspecified`,
    and all 37 section_headers have zero children. So the section path still
    comes from the heading-hierarchy rebuild, tracked here in document order.

    The section_path is prepended to each sub-chunk's text: without it a bare
    "- Machine Learning (Stanford University)" reranks poorly for a query like
    "certifications and qualifications".
    """
    out: list[MdChunk] = list(chunks)          # every parent survives, untouched
    if doc is None:
        for i, c in enumerate(out):
            c.chunk_index = i
        return out

    from docling_core.types.doc import DocItemLabel

    ancestors = heading_ancestors or {}
    current: str | None = None
    n_bullets = 0

    for item, _level in doc.iterate_items():
        label = getattr(item, 'label', None)
        text = (getattr(item, 'text', '') or '').strip()

        # Track the most recent heading in document order. The tree is flat, so
        # position is the only signal for which section a bullet belongs to.
        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            if text:
                current = text
            continue

        if label != DocItemLabel.LIST_ITEM or not text:
            continue

        chain = (ancestors.get(current, []) + [current]) if current else []
        section_path = ' > '.join(chain) or '(root)'

        regions = []
        for p in (getattr(item, 'prov', None) or []):
            page = getattr(p, 'page_no', None)
            bb = getattr(p, 'bbox', None)
            if page is None or bb is None:
                continue
            try:
                regions.append({'page_no': int(page),
                                'bbox': [float(v) for v in bb.as_tuple()]})
            except Exception:
                continue
        if not regions:
            # No geometry means no highlight target, which is the entire point
            # of a sub-chunk. The parent still carries the text.
            continue

        ctx_text = f'{section_path}\n- {text}' if section_path != '(root)' else f'- {text}'
        out.append(MdChunk(
            doc_path=rel_path or (chunks[0].doc_path if chunks else ''),
            chunk_index=0,                      # renumbered below
            section_path=section_path,
            text=ctx_text,
            token_count=tokenizer.count_tokens(ctx_text),
            page_no=regions[0]['page_no'],
            bbox=regions[0]['bbox'],
            regions=regions,
            section_level=len(chain) or None,
            kind='text',
            caption_for=None,
        ))
        n_bullets += 1

    logger.info('bullet sub-chunks from the document model: %d', n_bullets)
    for i, c in enumerate(out):
        c.chunk_index = i
    return out


# ── Caption manifest helpers ─────────────────────────────────────────


def _safe_progress(writer: Callable[[dict], None] | None, payload: dict) -> None:
    if writer is None:
        return
    try:
        writer(payload)
    except Exception:
        logger.exception('progress_writer raised; ignoring')


# ── Caption/vision pass: intentionally not ported ─────────────────────
# noted gated its picture/table captioning behind ENABLE_DOC_DESCRIPTIONS,
# which was OFF for the CV corpus, so the 69 live chunks never carried
# LLM captions. The CV is prose + bullets with no figures or tables worth
# describing, so the vision path would only add a model dependency for a
# code path that produces nothing. These two stubs keep the chunk loop's
# call sites intact.
def _build_caption_manifest(doc, rel_path, progress_writer=None) -> dict[str, dict[str, Any]]:
    return {}


def _resolve_chunk_caption(doc_items: list,
                           manifest: dict[str, dict[str, Any]]) -> tuple[str, None, None]:
    return 'text', None, None
