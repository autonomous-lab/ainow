#!/usr/bin/env node
/**
 * Memory DB CLI - Direct mode (no daemon)
 * Loads embedding model and SQLite directly in process
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from '@huggingface/transformers';
import * as sqliteVec from 'sqlite-vec';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = path.join(__dirname, '.cache/models');
const EMBEDDING_DIM = 384;
// Pin memory.db to the agent root (two levels up from this script) regardless
// of how the script is invoked (e.g. `cd .skills/memory-db && node cli.js ...`
// would otherwise look in the wrong place). MEMORY_DB_PATH overrides if set.
const DB_PATH = process.env.MEMORY_DB_PATH || path.resolve(__dirname, '..', '..', 'memory.db');

// Globals
let embeddingPipeline = null;
let db = null;
let queryCache = new Map();
const CACHE_MAX_SIZE = 100;

// Parse CLI args
function parseArgs(argsSlice) {
    const result = { _: [] };
    let i = 0;
    while (i < argsSlice.length) {
        if (argsSlice[i].startsWith('--')) {
            const key = argsSlice[i].slice(2);
            const value = argsSlice[i + 1] && !argsSlice[i + 1].startsWith('--') ? argsSlice[++i] : true;
            result[key] = value;
        } else {
            result._.push(argsSlice[i]);
        }
        i++;
    }
    return result;
}

// Load embedding model
async function loadModel() {
    if (embeddingPipeline) return embeddingPipeline;

    embeddingPipeline = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        { cache_dir: CACHE_DIR, dtype: 'fp32' }
    );
    return embeddingPipeline;
}

// Load SQLite database
async function loadDatabase() {
    if (db) return db;

    const { default: Database } = await import('better-sqlite3');
    db = new Database(DB_PATH);
    sqliteVec.load(db);

    // Initialize schema
    db.exec(`
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            type TEXT NOT NULL,
            interlocutor TEXT,
            language TEXT DEFAULT 'fr',
            content TEXT NOT NULL,
            summary TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
        CREATE INDEX IF NOT EXISTS idx_memories_interlocutor ON memories(interlocutor);
        CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);

        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            relationship TEXT,
            notes TEXT,
            preferred_language TEXT DEFAULT 'en',
            last_contact TEXT,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            topic TEXT NOT NULL,
            position TEXT NOT NULL,
            context TEXT,
            confidence TEXT DEFAULT 'high'
        );
        CREATE INDEX IF NOT EXISTS idx_positions_topic ON positions(topic);

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT NOT NULL,
            impact TEXT,
            related_people TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
    `);

    // Vector index (sqlite-vec): rowid is paired with memories.id
    db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[${EMBEDDING_DIM}])`);

    // One-time backfill: migrate old embeddings stored in memories.embedding BLOB
    // (legacy @sqliteai/sqlite-vector schema) into vec_memories.
    backfillLegacyEmbeddings(db);

    return db;
}

function backfillLegacyEmbeddings(db) {
    const cols = db.prepare(`PRAGMA table_info(memories)`).all();
    if (!cols.some(c => c.name === 'embedding')) return;

    const rows = db.prepare(`
        SELECT m.id, m.embedding
        FROM memories m
        WHERE m.embedding IS NOT NULL
          AND length(m.embedding) = ${EMBEDDING_DIM * 4}
          AND NOT EXISTS (SELECT 1 FROM vec_memories v WHERE v.rowid = m.id)
    `).all();

    if (rows.length === 0) return;

    const insertVec = db.prepare(`INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)`);
    const tx = db.transaction((batch) => {
        for (const r of batch) {
            insertVec.run(BigInt(r.id), r.embedding);
        }
    });
    tx(rows);
    console.error(`[memory-db] Backfilled ${rows.length} legacy embeddings into vec_memories`);
}

// Convert a JS number array to a Float32 buffer (sqlite-vec native binding)
function embeddingBuffer(embedding) {
    return Buffer.from(new Float32Array(embedding).buffer);
}

// Generate embedding with caching
async function generateEmbedding(text) {
    const truncated = text.length > 1000 ? text.substring(0, 1000) : text;

    if (queryCache.has(truncated)) {
        return queryCache.get(truncated);
    }

    const extractor = await loadModel();
    const output = await extractor(truncated, {
        pooling: 'mean',
        normalize: true
    });

    const embedding = Array.from(new Float32Array(output.data));

    // LRU cache
    if (queryCache.size >= CACHE_MAX_SIZE) {
        const firstKey = queryCache.keys().next().value;
        queryCache.delete(firstKey);
    }
    queryCache.set(truncated, embedding);

    return embedding;
}

function embeddingToJson(embedding) {
    return '[' + embedding.join(',') + ']';
}

// Session transcript search
const SESSION_FILE = path.join(process.cwd(), '.sessionIds.json');

function searchSessionTranscripts(query, limit = 10) {
    if (!fs.existsSync(SESSION_FILE)) return [];

    let data;
    try {
        data = JSON.parse(fs.readFileSync(SESSION_FILE, 'utf-8'));
    } catch (e) { return []; }

    const sessions = data.sessions || [];
    if (sessions.length === 0) return [];

    const searchTerm = query.toLowerCase();
    const matches = [];
    const MAX_PER_SESSION = 3;

    for (const session of sessions) {
        if (matches.length >= limit) break;
        if (!session.transcript_path || !fs.existsSync(session.transcript_path)) continue;

        let content;
        try {
            content = fs.readFileSync(session.transcript_path, 'utf-8');
        } catch (e) { continue; }

        const lines = content.split('\n').filter(l => l.trim());
        let sessionMatches = 0;

        for (const line of lines) {
            if (matches.length >= limit || sessionMatches >= MAX_PER_SESSION) break;
            try {
                const entry = JSON.parse(line);
                if (entry.type !== 'user' && entry.type !== 'assistant') continue;

                let text = '';
                if (entry.type === 'user') {
                    if (typeof entry.message === 'string') {
                        text = entry.message;
                    } else if (entry.message && entry.message.content) {
                        if (Array.isArray(entry.message.content)) {
                            text = entry.message.content
                                .filter(b => b.type === 'text')
                                .map(b => b.text)
                                .join(' ');
                        } else if (typeof entry.message.content === 'string') {
                            text = entry.message.content;
                        }
                    }
                } else {
                    const mc = entry.message && entry.message.content;
                    if (Array.isArray(mc)) {
                        text = mc
                            .filter(b => b.type === 'text')
                            .map(b => b.text)
                            .filter(t => t.trim() !== '(no content)')
                            .join(' ');
                    } else if (typeof mc === 'string') {
                        text = mc;
                    }
                }

                if (text && text.toLowerCase().includes(searchTerm)) {
                    const idx = text.toLowerCase().indexOf(searchTerm);
                    const start = Math.max(0, idx - 100);
                    const end = Math.min(text.length, idx + searchTerm.length + 100);
                    const snippet = (start > 0 ? '...' : '') +
                        text.slice(start, end) +
                        (end < text.length ? '...' : '');

                    matches.push({
                        sessionId: session.id,
                        sessionTitle: session.title || 'Untitled',
                        role: entry.type,
                        content: snippet,
                        timestamp: session.updated_at || session.created_at
                    });
                    sessionMatches++;
                }
            } catch (e) { /* skip bad line */ }
        }
    }

    return matches;
}

function printTranscriptResults(results) {
    if (results.length === 0) return;
    console.log(`\n━━━ Session Transcripts (${results.length} matches) ━━━\n`);
    results.forEach((r, i) => {
        console.log(`[${i + 1}] Session: "${r.sessionTitle}" | ${r.role} | ${r.timestamp || 'unknown date'}`);
        console.log(`    ${r.content.substring(0, 300)}${r.content.length > 300 ? '...' : ''}`);
        console.log('');
    });
}

// Format memory output
function printMemories(memories, label = 'memories') {
    if (memories.length === 0) {
        console.log(`No ${label} found.`);
        return;
    }
    console.log(`Found ${memories.length} ${label}:\n`);
    memories.forEach((r, i) => {
        console.log(`[${i + 1}] ID: ${r.id} | Type: ${r.type} | ${r.created_at}`);
        if (r.interlocutor) console.log(`    Person: ${r.interlocutor}`);
        console.log(`    Content: ${r.content.substring(0, 2000)}${r.content.length > 2000 ? '...' : ''}`);
        if (r.distance !== undefined) console.log(`    Distance: ${r.distance?.toFixed(4) || 'N/A'}`);
        console.log('');
    });
}

// Main
const args = process.argv.slice(2);
const command = args[0];
const options = parseArgs(args.slice(1));

// HELP
if (!command || command === 'help') {
    console.log('Memory DB - Persistent memory with semantic search (direct mode)');
    console.log('Commands: remember, recall, search, recent, get, delete, person, stats');
    process.exit(0);
}

// Initialize - load model FIRST (before SQLite to avoid ONNX conflicts)
await loadModel();
await loadDatabase();

// RECALL - semantic search (single or multi)
if (command === 'recall') {
    const queryArg = options._.join(' ') || options.query;
    if (!queryArg) {
        console.error('Error: Query is required. Usage: recall "query1" "query2" ... or recall "query1|query2|..."');
        process.exit(1);
    }

    // Check for multi-query: either multiple args or pipe-separated
    const queries = options._.length > 1
        ? options._
        : queryArg.includes('|')
            ? queryArg.split('|').map(q => q.trim()).filter(q => q)
            : [queryArg];

    // Detect if query asks for recent/latest memories (multi-language support)
    // French, English, Spanish, German, Italian, Portuguese, Dutch, Polish, Russian, Hebrew, Arabic, Chinese, Japanese, Korean, Turkish
    const recentKeywords = new RegExp([
        // French
        'dernier', 'derniers', 'dernière', 'dernières', 'récent', 'récents', 'récente', 'récentes',
        'nouveau', 'nouveaux', 'nouvelle', 'nouvelles', 'hier', 'aujourd\'hui', 'cette semaine',
        // English
        'recent', 'recently', 'latest', 'last', 'fresh', 'new', 'newest', 'today', 'yesterday', 'this week',
        // Spanish
        'reciente', 'recientes', 'último', 'últimos', 'última', 'últimas', 'nuevo', 'nuevos', 'nueva', 'nuevas',
        'hoy', 'ayer', 'esta semana',
        // German
        'letzt', 'letzte', 'letzter', 'letztes', 'letzten', 'neu', 'neue', 'neuer', 'neues', 'neuen',
        'neueste', 'neuesten', 'heute', 'gestern', 'diese woche', 'kürzlich', 'aktuell',
        // Italian
        'recente', 'recenti', 'ultimo', 'ultimi', 'ultima', 'ultime', 'nuovo', 'nuovi', 'nuova', 'nuove',
        'oggi', 'ieri', 'questa settimana',
        // Portuguese
        'recente', 'recentes', 'último', 'últimos', 'última', 'últimas', 'novo', 'novos', 'nova', 'novas',
        'hoje', 'ontem', 'esta semana',
        // Dutch
        'laatste', 'recent', 'recente', 'nieuw', 'nieuwe', 'vandaag', 'gisteren', 'deze week',
        // Polish
        'ostatni', 'ostatnia', 'ostatnie', 'niedawny', 'niedawna', 'nowy', 'nowa', 'nowe', 'dziś', 'wczoraj',
        // Russian
        'последний', 'последняя', 'последние', 'недавний', 'недавняя', 'недавние', 'новый', 'новая', 'новые',
        'сегодня', 'вчера',
        // Hebrew
        'אחרון', 'אחרונה', 'אחרונים', 'אחרונות', 'חדש', 'חדשה', 'חדשים', 'חדשות', 'היום', 'אתמול',
        // Arabic
        'الأخير', 'الأخيرة', 'جديد', 'جديدة', 'اليوم', 'أمس',
        // Chinese
        '最近', '最新', '最后', '新的', '今天', '昨天', '这周', '本周', '近期',
        // Japanese
        '最近', '最新', '最後', '新しい', '今日', '昨日', '今週',
        // Korean
        '최근', '최신', '마지막', '새로운', '오늘', '어제', '이번주',
        // Turkish
        'son', 'yeni', 'bugün', 'dün', 'bu hafta'
    ].join('|'), 'i');
    const wantsRecent = queries.some(q => recentKeywords.test(q)) || options.sort === 'date';

    if (wantsRecent) {
        console.log('(Sorting by date: most recent first)\n');
    }

    if (queries.length === 1) {
        // Single query
        const limit = parseInt(options.limit) || 10;
        const embedding = await generateEmbedding(queries[0]);
        const queryBuf = embeddingBuffer(embedding);

        const vectorResults = db.prepare(
            `SELECT rowid, distance FROM vec_memories WHERE embedding MATCH ? AND k = 30 ORDER BY distance`
        ).all(queryBuf);

        let resultsWithDistance = [];

        if (vectorResults.length > 0) {
            const rowids = vectorResults.map(r => r.rowid);
            const distanceMap = new Map(vectorResults.map(r => [r.rowid, r.distance]));

            let sql = `SELECT * FROM memories WHERE id IN (${rowids.map(() => '?').join(',')})`;
            const params = [...rowids];

            if (options.type) { sql += ` AND type = ?`; params.push(options.type); }
            if (options.interlocutor) { sql += ` AND interlocutor = ?`; params.push(options.interlocutor); }

            const results = db.prepare(sql).all(...params);
            resultsWithDistance = results.map(r => ({
                ...r,
                distance: distanceMap.get(r.id),
                embedding: undefined
            })).sort((a, b) => {
                if (wantsRecent) {
                    // Sort by date (most recent first)
                    return new Date(b.created_at) - new Date(a.created_at);
                }
                // Sort by relevance (closest distance first)
                return a.distance - b.distance;
            }).slice(0, limit);
        }

        if (resultsWithDistance.length === 0) {
            console.log('No memories found.');
        } else {
            printMemories(resultsWithDistance);
        }

        // Always search session transcripts as complement (not just fallback)
        if (options['no-transcripts'] !== true) {
            const transcriptResults = searchSessionTranscripts(queries[0], limit);
            if (transcriptResults.length > 0) {
                printTranscriptResults(transcriptResults);
            } else if (resultsWithDistance.length === 0) {
                console.log('No matches in session transcripts either.');
            }
        }
    } else {
        // Multi query
        const limit = parseInt(options.limit) || 5;
        const allResults = {};
        const start = Date.now();

        for (const query of queries) {
            const embedding = await generateEmbedding(query);
            const queryJson = embeddingToJson(embedding);

            const vectorResults = db.prepare(
                `SELECT rowid, distance FROM vector_quantize_scan('memories', 'embedding', vector_as_f32(?), 30)`
            ).all(queryJson);

            if (vectorResults.length === 0) {
                allResults[query] = [];
                continue;
            }

            const rowids = vectorResults.map(r => r.rowid);
            const distanceMap = new Map(vectorResults.map(r => [r.rowid, r.distance]));

            const sql = `SELECT * FROM memories WHERE id IN (${rowids.map(() => '?').join(',')})`;
            const memories = db.prepare(sql).all(...rowids);

            allResults[query] = memories.map(r => ({
                ...r,
                distance: distanceMap.get(r.id),
                embedding: undefined
            })).sort((a, b) => {
                if (wantsRecent) {
                    // Sort by date (most recent first)
                    return new Date(b.created_at) - new Date(a.created_at);
                }
                // Sort by relevance (closest distance first)
                return a.distance - b.distance;
            }).slice(0, limit);
        }

        console.log(`Found results for ${queries.length} queries (${Date.now() - start}ms total):\n`);

        for (const [query, memories] of Object.entries(allResults)) {
            console.log(`━━━ "${query}" (${memories.length} results) ━━━`);
            if (memories.length === 0) {
                console.log('  No memories found.\n');
            } else {
                memories.forEach((r, i) => {
                    console.log(`[${i + 1}] ID: ${r.id} | ${r.type} | ${r.created_at}`);
                    if (r.interlocutor) console.log(`    Person: ${r.interlocutor}`);
                    console.log(`    ${r.content.substring(0, 200)}${r.content.length > 200 ? '...' : ''}`);
                    console.log(`    Distance: ${r.distance?.toFixed(4) || 'N/A'}\n`);
                });
            }
        }

        // Always search session transcripts as complement
        if (options['no-transcripts'] !== true) {
            for (const [query] of Object.entries(allResults)) {
                const transcriptResults = searchSessionTranscripts(query, limit);
                if (transcriptResults.length > 0) {
                    console.log(`━━━ "${query}" - Session Transcripts (${transcriptResults.length} matches) ━━━`);
                    transcriptResults.forEach((r, i) => {
                        console.log(`[${i + 1}] Session: "${r.sessionTitle}" | ${r.role} | ${r.timestamp || 'unknown'}`);
                        console.log(`    ${r.content.substring(0, 200)}${r.content.length > 200 ? '...' : ''}\n`);
                    });
                }
            }
        }
    }
    process.exit(0);
}

// REMEMBER - store memory
if (command === 'remember') {
    if (!options.type || !options.content) {
        console.error('Error: --type and --content are required');
        console.error('Usage: remember --type TYPE --content "content"');
        console.error('Multi:  remember --type TYPE --content "content1|||content2|||content3"');
        process.exit(1);
    }

    // Check for multi-content
    const contents = options.content.includes('|||')
        ? options.content.split('|||').map(c => c.trim()).filter(c => c)
        : [options.content];

    const insertMemory = db.prepare(`
        INSERT INTO memories (type, content, interlocutor, language, summary, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    `);
    const insertVector = db.prepare(`
        INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)
    `);

    const ids = [];
    for (const content of contents) {
        const textToEmbed = options.summary || content;
        const embedding = await generateEmbedding(textToEmbed);
        const embBuf = embeddingBuffer(embedding);

        const result = db.transaction(() => {
            const r = insertMemory.run(
                options.type,
                content,
                options.interlocutor || null,
                options.language || 'fr',
                options.summary || null,
                JSON.stringify(options.metadata ? JSON.parse(options.metadata) : {})
            );
            // sqlite-vec vec0 requires BigInt rowids
            insertVector.run(BigInt(r.lastInsertRowid), embBuf);
            return r;
        })();
        ids.push(result.lastInsertRowid);
    }

    if (ids.length === 1) {
        console.log(`Memory stored with ID: ${ids[0]}`);
    } else {
        console.log(`${ids.length} memories stored with IDs: ${ids.join(', ')}`);
    }
    process.exit(0);
}

// SEARCH - keyword search
if (command === 'search') {
    const query = options._.join(' ') || options.query;
    if (!query) {
        console.error('Error: Query is required');
        process.exit(1);
    }

    let sql = `SELECT * FROM memories WHERE content LIKE ?`;
    const params = [`%${query}%`];

    if (options.type) { sql += ` AND type = ?`; params.push(options.type); }
    if (options.interlocutor) { sql += ` AND interlocutor = ?`; params.push(options.interlocutor); }
    sql += ` ORDER BY created_at DESC LIMIT ?`;
    params.push(parseInt(options.limit) || 20);

    const results = db.prepare(sql).all(...params).map(r => ({
        ...r,
        embedding: undefined
    }));

    printMemories(results);
    process.exit(0);
}

// RECENT
if (command === 'recent') {
    let sql = `SELECT * FROM memories WHERE 1=1`;
    const params = [];

    if (options.type) { sql += ` AND type = ?`; params.push(options.type); }
    if (options.interlocutor) { sql += ` AND interlocutor = ?`; params.push(options.interlocutor); }
    sql += ` ORDER BY created_at DESC LIMIT ?`;
    params.push(parseInt(options.limit) || 10);

    const results = db.prepare(sql).all(...params).map(r => ({
        ...r,
        embedding: undefined
    }));

    printMemories(results, 'recent memories');
    process.exit(0);
}

// GET
if (command === 'get') {
    if (!options.id) {
        console.error('Error: --id is required');
        process.exit(1);
    }

    const memory = db.prepare(`SELECT * FROM memories WHERE id = ?`).get(parseInt(options.id));

    if (!memory) {
        console.log(`Memory ${options.id} not found.`);
    } else {
        console.log(`Memory ID: ${memory.id}`);
        console.log(`Type: ${memory.type}`);
        console.log(`Created: ${memory.created_at}`);
        if (memory.interlocutor) console.log(`Person: ${memory.interlocutor}`);
        if (memory.language) console.log(`Language: ${memory.language}`);
        if (memory.summary) console.log(`Summary: ${memory.summary}`);
        console.log(`\nFull Content:\n${'-'.repeat(80)}\n${memory.content}\n${'-'.repeat(80)}`);
    }
    process.exit(0);
}

// DELETE
if (command === 'delete') {
    if (!options.id) {
        console.error('Error: --id is required');
        process.exit(1);
    }

    const memId = parseInt(options.id);
    db.transaction(() => {
        db.prepare(`DELETE FROM memories WHERE id = ?`).run(memId);
        db.prepare(`DELETE FROM vec_memories WHERE rowid = ?`).run(BigInt(memId));
    })();
    console.log(`Memory ${options.id} deleted.`);
    process.exit(0);
}

// PERSON
if (command === 'person') {
    const subcommand = options._[0];

    if (subcommand === 'add') {
        if (!options.name) {
            console.error('Error: --name is required');
            process.exit(1);
        }
        db.prepare(`
            INSERT INTO people (name, relationship, notes, preferred_language, last_contact, metadata)
            VALUES (?, ?, ?, ?, datetime('now'), ?)
            ON CONFLICT(name) DO UPDATE SET
                relationship = excluded.relationship,
                notes = excluded.notes,
                preferred_language = excluded.preferred_language,
                last_contact = datetime('now'),
                metadata = excluded.metadata
        `).run(
            options.name,
            options.relationship || null,
            options.notes || null,
            options.language || 'en',
            JSON.stringify(options.metadata ? JSON.parse(options.metadata) : {})
        );
        console.log(`Person "${options.name}" saved.`);
    } else if (subcommand === 'get') {
        const person = db.prepare(`SELECT * FROM people WHERE name = ?`).get(options.name);
        if (person) {
            person.metadata = JSON.parse(person.metadata || '{}');
            console.log(JSON.stringify(person, null, 2));
        } else {
            console.log(`Person "${options.name}" not found.`);
        }
    } else if (subcommand === 'list') {
        const people = db.prepare(`SELECT * FROM people ORDER BY last_contact DESC`).all();
        if (people.length === 0) {
            console.log('No people recorded yet.');
        } else {
            console.log(`${people.length} people:\n`);
            people.forEach(p => {
                console.log(`- ${p.name} (${p.relationship || 'unknown'}) - Last contact: ${p.last_contact || 'never'}`);
                if (p.notes) console.log(`  Notes: ${p.notes}`);
            });
        }
    } else {
        console.log('Usage: person <add|get|list> [options]');
    }
    process.exit(0);
}

// STATS
if (command === 'stats') {
    const memoriesCount = db.prepare(`SELECT COUNT(*) as count FROM memories`).get();
    const peopleCount = db.prepare(`SELECT COUNT(*) as count FROM people`).get();
    const positionsCount = db.prepare(`SELECT COUNT(*) as count FROM positions`).get();
    const eventsCount = db.prepare(`SELECT COUNT(*) as count FROM events`).get();
    const typeBreakdown = db.prepare(`SELECT type, COUNT(*) as count FROM memories GROUP BY type`).all();

    console.log('Memory Statistics:');
    console.log('==================');
    console.log(`Total memories: ${memoriesCount.count}`);
    console.log(`People tracked: ${peopleCount.count}`);
    console.log(`Positions recorded: ${positionsCount.count}`);
    console.log(`Events logged: ${eventsCount.count}`);
    if (typeBreakdown.length > 0) {
        console.log('\nMemories by type:');
        typeBreakdown.forEach(t => console.log(`  ${t.type}: ${t.count}`));
    }
    process.exit(0);
}

// Legacy daemon command - just inform user
if (command === 'daemon' || command === 'start-daemon') {
    console.log('Daemon mode has been removed. The CLI now runs directly without a daemon.');
    console.log('All commands work the same way, just without needing to start/stop a daemon.');
    process.exit(0);
}

console.error(`Unknown command: ${command}`);
console.log('Use "help" to see available commands.');
process.exit(1);
