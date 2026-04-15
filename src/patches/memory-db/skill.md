# Memory DB Skill

Persistent memory system with semantic search using SQLite and local embeddings.

## Features

- **Semantic Search**: Find memories by meaning using vector similarity
- **Keyword Search**: Traditional text search with LIKE patterns
- **People Tracking**: Record information about people you interact with
- **Local Embeddings**: Uses all-MiniLM-L6-v2 model (384 dimensions) - runs completely offline
- **Vector Storage**: sqlite-vector extension for efficient similarity search
- **Direct Mode**: No daemon required, loads model directly in process

## Requirements

- **Node.js**
- `@huggingface/transformers` for local embedding generation
- `better-sqlite3` for SQLite database
- `@sqliteai/sqlite-vector-linux-x86_64` for vector operations

## Usage

### Store a memory

```bash
# Single memory
node cli.js remember --type <type> --content "<content>" [options]

# Multiple memories at once (triple pipe separator)
node cli.js remember --type <type> --content "content1|||content2|||content3"
```

Options:
- `--type` (required): Memory type (conversation, decision, fact, insight, war-story, tech-opinion, business-metric, current-focus)
- `--content` (required): The content to remember (use `|||` to separate multiple)
- `--interlocutor`: Person involved
- `--summary`: Short summary for embedding (if content is long)
- `--language`: Language code (default: fr)

Examples:
```bash
node cli.js remember --type conversation --content "Discussed backup strategy with team" --interlocutor Amit

# Store multiple facts at once
node cli.js remember --type fact --content "Fact one|||Fact two|||Fact three"
```

### Semantic search (recall)

```bash
# Single query
node cli.js recall "<search query>" [--limit N] [--type TYPE] [--interlocutor NAME]

# Multiple queries (pipe-separated) - efficient batch search!
node cli.js recall "query1|query2|query3" [--limit N]

# Multiple queries (separate arguments)
node cli.js recall "query1" "query2" "query3" [--limit N]
```

Finds memories semantically similar to your query. Lower distance = more similar.

**Session transcript search:** Recall always searches through session transcript JSONL files listed in `.sessionIds.json` in addition to memory. This means even if something was discussed but never stored in memory, recall will find it in past conversation history. Transcript results are shown separately under "Session Transcripts".

To disable transcript search: `--no-transcripts`

Examples:
```bash
node cli.js recall "backup discussions"
node cli.js recall "team meetings" --limit 5 --type conversation

# Search multiple topics at once
node cli.js recall "GetATeam|AppDrag|Benjamin|Steven" --limit 3
node cli.js recall "Elestio MRR" "David Alimi" "DDoS attack" --limit 2

# Disable session transcript search
node cli.js recall "some query" --no-transcripts
```

### Keyword search

```bash
node cli.js search "<keyword>" [--limit N] [--type TYPE]
```

Traditional text search - finds exact matches in content.

### Recent memories

```bash
node cli.js recent [--limit N] [--type TYPE]
```

### Get full memory content

```bash
node cli.js get --id <ID>
```

### Delete a memory

```bash
node cli.js delete --id <ID>
```

### Manage people

```bash
# Add/update a person
node cli.js person add --name "Amit" --relationship "colleague" --notes "Support lead"

# Get person details
node cli.js person get --name "Amit"

# List all people
node cli.js person list [--relationship TYPE]
```

### View statistics

```bash
node cli.js stats
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   CLI (Node)                     │
│                                                  │
│   ┌───────────────┐    ┌──────────────────────┐ │
│   │  Embedding    │    │      SQLite DB       │ │
│   │  Model (ONNX) │    │  + Vector Extension  │ │
│   │  MiniLM-L6-v2 │    │    memory.db         │ │
│   └───────────────┘    └──────────────────────┘ │
│                                                  │
│   - Model loaded on first use                    │
│   - Cached in .cache/models                      │
│   - LRU cache for 100 embeddings                 │
└─────────────────────────────────────────────────┘
```

## Database Location

The `memory.db` file is stored in the **agent's data directory** (current working directory), not in the skill folder. This allows each agent to have its own isolated memory database.

```
/opt/app/data/           <- Agent's directory
├── memory.db            <- Memory database (stored HERE)
├── CLAUDE.md
└── .skills/
    └── memory-db/       <- Skill folder (code only)
        ├── cli.js
        └── ...
```

**Important:** Always run the CLI from the agent's directory to ensure the correct memory.db is used.

## Database Schema

The database includes tables for:
- `memories`: Main memory storage with embeddings
- `people`: Contact information
- `positions`: Your positions/opinions on topics
- `events`: Important events to remember

## Memory Types

Recommended types for consistency:
- `conversation` - Discussions with specific people
- `decision` - Decisions made and rationale
- `fact` - Factual information to remember
- `insight` - Learnings and realizations
- `war-story` - Notable incidents/experiences
- `tech-opinion` - Technical positions/preferences
- `business-metric` - Business data points
- `current-focus` - Current priorities and context

## Technical Notes

### ONNX/sqlite-vector Conflict

There's a known conflict between the ONNX runtime (used by transformers.js) and the sqlite-vector native extension. This is handled by:

1. Loading the embedding model FIRST (before any SQLite operations)
2. Generating embeddings BEFORE database queries

### Embedding Model

Uses Xenova/all-MiniLM-L6-v2 which produces 384-dimension vectors. Good for sentences and small paragraphs (up to ~256 tokens). Texts longer than 1000 characters are truncated.

First run may take a moment to download the model (~23MB), which is cached in `.cache/models`.

### Embedding Cache

The CLI maintains an LRU cache of 100 recent query embeddings in memory. Repeated searches for the same text within a session are instant.
